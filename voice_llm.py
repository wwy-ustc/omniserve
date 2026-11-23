"""Local Qwen model loading and single-request streaming generation."""

from collections.abc import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_budget import cuda_admission_budget, estimate_kv_bytes
from workers.llm_worker import MODEL_ID, MODEL_REVISION


class VoiceLanguageModel:
    """Keep weights loaded while each request owns its own KV cache."""

    def __init__(self, device: str, local_files_only: bool, max_new_tokens: int):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"Unknown LLM device: {device}")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is unavailable.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CUDA is unavailable.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.dtype = torch.float32 if device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=local_files_only,
            dtype=self.dtype,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=local_files_only,
        )
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
        self.model_id = MODEL_ID
        self.revision = MODEL_REVISION

    def _inputs(self, prompt: str):
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        context_tokens = inputs["input_ids"].shape[-1] + self.max_new_tokens
        if context_tokens > self.model.config.max_position_embeddings:
            raise ValueError("The transcript and output budget exceed the Qwen context limit.")
        return inputs

    def prompt_token_ids(self, prompt: str) -> list[int]:
        return self._inputs(prompt)["input_ids"][0].tolist()

    def visible_text(self, token_ids: list[int]) -> str:
        # Byte-level tokenization can split a Unicode character across tokens.
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).rstrip("\ufffd")

    def _budget(self, input_tokens: int) -> dict:
        config = self.model.config
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        kv_bytes = estimate_kv_bytes(
            layers=config.num_hidden_layers, kv_heads=config.num_key_value_heads,
            head_dim=head_dim, tokens=input_tokens + self.max_new_tokens,
            element_bytes=self.dtype.itemsize,
        )
        budget = {
            "input_tokens": input_tokens,
            "max_new_tokens": self.max_new_tokens,
            "estimated_kv_bytes": kv_bytes,
            "cuda_free_bytes": None,
            "cuda_required_bytes": None,
        }
        if self.device == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            budget.update(cuda_admission_budget(
                kv_bytes=kv_bytes, input_tokens=input_tokens,
                attention_heads=config.num_attention_heads,
                driver_free_bytes=free_bytes, total_bytes=total_bytes,
                reserved_bytes=torch.cuda.memory_reserved(self.device),
                allocated_bytes=torch.cuda.memory_allocated(self.device),
            ))
        return budget

    def fit(self, prompt: str) -> dict:
        """Estimate per-request memory and report the available CUDA admission budget."""
        input_tokens = self._inputs(prompt)["input_ids"].shape[-1]
        return self._budget(input_tokens)

    def peak_cuda_bytes(self) -> int | None:
        return torch.cuda.max_memory_allocated() if self.device == "cuda" else None

    @torch.inference_mode()
    def generate(
        self, prompt: str, on_text: Callable[[str], None], *, enforce_budget: bool = False,
        on_complete: Callable[[list[int]], None] | None = None,
    ) -> str:
        """Emit cumulative Unicode-safe text as tokens arrive, then return the full answer."""
        inputs = self._inputs(prompt)
        if enforce_budget:
            budget = self._budget(inputs["input_ids"].shape[-1])
            required = budget["cuda_required_bytes"]
            free = budget["cuda_free_bytes"]
            if required is not None and required > free:
                raise RuntimeError(
                    f"CUDA memory budget exceeded ({required / 2**20:.0f} MiB required, "
                    f"{free / 2**20:.0f} MiB free)."
                )

        inputs = inputs.to(self.device)
        mask = inputs["attention_mask"]
        input_tokens = mask.shape[-1]
        full_mask = torch.cat((mask, mask.new_ones((1, self.max_new_tokens))), dim=-1)
        outputs = self.model(**inputs, use_cache=True, logits_to_keep=1)
        generated = []
        visible = ""
        for step in range(self.max_new_tokens):
            token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = token.item()
            generated.append(token_id)
            if token_id in self.eos_ids:
                break
            candidate = self.visible_text(generated)
            if candidate != visible:
                visible = candidate
                on_text(visible)
            if step + 1 < self.max_new_tokens:
                outputs = self.model(
                    input_ids=token,
                    attention_mask=full_mask[:, :input_tokens + step + 1],
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
        if on_complete is not None:
            on_complete(generated)
        return visible
