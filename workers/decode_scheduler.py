"""Per-request Qwen KV state and one shared decode forward per step."""

from dataclasses import dataclass, field

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer


@dataclass(frozen=True)
class Request:
    arrival: int
    input_ids: list[int]
    max_new_tokens: int


@dataclass
class State:
    request: Request
    generated: list[int] = field(default_factory=list)
    cache: DynamicCache | None = None


class Scheduler:
    """FIFO admission with separate prefills and one shared decode per tick."""

    def __init__(self, model, device, max_active):
        if max_active < 1:
            raise ValueError("max_active must be positive.")
        if set(model.config.get_text_config().layer_types) != {"full_attention"}:
            raise ValueError("This scheduler requires full-attention KV cache layers.")
        self.model = model
        self.device = device
        self.max_active = max_active
        eos = model.generation_config.eos_token_id
        self.stops = {eos} if isinstance(eos, int) else set(eos or ())

    @torch.inference_mode()
    def start(self, request):
        if request.arrival < 0 or request.max_new_tokens < 1 or not request.input_ids:
            raise ValueError("Requests need a nonnegative arrival, prompt tokens and a positive token limit.")
        state = State(request)
        self._prefill(state)
        if self.finished(state):
            state.cache = None
        return state

    def _prefill(self, state):
        ids = torch.tensor([state.request.input_ids], device=self.device)
        output = self.model(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            use_cache=True, logits_to_keep=1,
        )
        state.cache = output.past_key_values
        if not all(type(layer) is DynamicLayer for layer in state.cache.layers):
            raise RuntimeError("Expected DynamicLayer caches from Qwen.")
        state.generated.append(output.logits[0, -1].argmax().item())

    @torch.inference_mode()
    def decode(self, states):
        # The most recently emitted token is pending and has no KV entry yet.
        lengths = [len(s.request.input_ids) + len(s.generated) - 1 for s in states]
        if any(s.cache.get_seq_length() != n for s, n in zip(states, lengths, strict=True)):
            raise RuntimeError("Per-request cache length does not match the token history.")
        width = max(lengths)
        lengths_tensor = torch.tensor(lengths, device=self.device)
        columns = torch.arange(width + 1, device=self.device)
        mask = columns[None, :] >= width - lengths_tensor[:, None]

        def pack(tensors):
            return torch.cat([
                torch.nn.functional.pad(tensor, (0, 0, width - length, 0))
                for tensor, length in zip(tensors, lengths, strict=True)
            ])

        if len(states) == 1:
            cache = states[0].cache
        else:
            cache = DynamicCache([
                (pack([layer.keys for layer in layers]), pack([layer.values for layer in layers]))
                for layers in zip(*(s.cache.layers for s in states), strict=True)
            ])
        output = self.model(
            input_ids=torch.tensor([[s.generated[-1]] for s in states], device=self.device),
            attention_mask=mask, position_ids=lengths_tensor[:, None],
            past_key_values=cache, use_cache=True, logits_to_keep=1,
        )
        if output.past_key_values.get_seq_length() != width + 1:
            raise RuntimeError("Decode forward did not append exactly one KV position.")
        tokens = output.logits[:, -1].argmax(dim=-1).tolist()
        for row, (state, length, token) in enumerate(zip(states, lengths, tokens, strict=True)):
            state.generated.append(token)
            if self.finished(state):
                state.cache = None
            elif len(states) == 1:
                state.cache = output.past_key_values
            else:
                # Remove left padding and avoid retaining another request's storage.
                start = width - length
                state.cache = DynamicCache([
                    (layer.keys[row:row + 1, :, start:, :].clone(),
                     layer.values[row:row + 1, :, start:, :].clone())
                    for layer in output.past_key_values.layers
                ])

        return lengths

    def finished(self, state):
        return (state.generated[-1] in self.stops
                or len(state.generated) >= state.request.max_new_tokens)
