"""Single-request, fixed-length Qwen prefill and decode benchmark."""

import argparse
import json
from statistics import fmean
from time import perf_counter_ns

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def resolve_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install CUDA-enabled PyTorch or use --device cpu.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use --device cuda or --device cpu.")
    return device


def device_metadata(device):
    if device != "cuda":
        return {}
    properties = torch.cuda.get_device_properties(device)
    return {
        "cuda_version": torch.version.cuda,
        "cuda_device_name": properties.name,
        "cuda_total_memory_mb": properties.total_memory / 2**20,
    }


def reset_peak_memory(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device):
    if device != "cuda":
        return {}
    return {
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
    }


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", help="Hugging Face revision or commit SHA")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--prompt", default="你是谁")
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--runs", type=positive_int, default=20)
    parser.add_argument("--decode-steps", type=positive_int, default=16)
    return parser.parse_args()


def load_model(args):
    args.device = resolve_device(args.device)
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    requested_revision = args.revision
    if requested_revision is None and args.model == MODEL_ID:
        requested_revision = MODEL_REVISION
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=requested_revision,
        local_files_only=args.local_files_only, dtype=dtype,
    ).to(args.device).eval()
    # Use the resolved weights commit for the tokenizer as well.
    revision = getattr(model.config, "_commit_hash", None) or requested_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=revision, local_files_only=args.local_files_only,
    )
    return model, tokenizer, revision


def prepare_inputs(tokenizer, prompt, device, decode_steps):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(device)
    mask = inputs["attention_mask"]
    full_mask = torch.cat((mask, mask.new_ones((1, decode_steps))), dim=-1)
    decode_masks = [
        full_mask[:, :mask.shape[-1] + step]
        for step in range(1, decode_steps + 1)
    ]
    return inputs, decode_masks


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def run_request(model, inputs, decode_masks, device):
    """Each call owns a fresh cache. Decode mutates it only within this call."""
    synchronize(device)
    reset_peak_memory(device)
    start = perf_counter_ns()
    outputs = model(**inputs, use_cache=True, logits_to_keep=1)
    synchronize(device)
    prefill_end = perf_counter_ns()

    token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    synchronize(device)
    first_token_end = perf_counter_ns()

    cache = outputs.past_key_values
    prefill_cache_length = int(cache.get_seq_length())
    tokens = [token]
    decode_start = perf_counter_ns()
    for mask in decode_masks:
        outputs = model(
            input_ids=token, attention_mask=mask, past_key_values=cache,
            use_cache=True, logits_to_keep=1,
        )
        cache = outputs.past_key_values
        token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(token)
    synchronize(device)
    decode_end = perf_counter_ns()

    input_length = inputs["input_ids"].shape[-1]
    final_cache_length = int(cache.get_seq_length())
    if (prefill_cache_length, final_cache_length) != (
        input_length, input_length + len(decode_masks)
    ):
        raise RuntimeError(
            f"Unexpected KV cache lengths: {prefill_cache_length} → {final_cache_length}"
        )

    decode_seconds = (decode_end - decode_start) / 1e9
    metrics = {
        "prefill_ms": (prefill_end - start) / 1e6,
        "ttft_ms": (first_token_end - start) / 1e6,
        "decode_ms": decode_seconds * 1000,
        "decode_tokens_per_second": len(decode_masks) / decode_seconds,
        **peak_memory(device),
    }
    token_ids = torch.cat(tokens, dim=-1)[0].tolist()
    return metrics, token_ids, (prefill_cache_length, final_cache_length)


def summarize(samples):
    ordered = sorted(samples)

    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {"mean": fmean(ordered), "p50": percentile(0.5), "p95": percentile(0.95)}


def main():
    args = parse_args()
    try:
        model, tokenizer, revision = load_model(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    inputs, decode_masks = prepare_inputs(
        tokenizer, args.prompt, args.device, args.decode_steps,
    )
    _, expected_tokens, _ = run_request(model, inputs, decode_masks, args.device)
    for _ in range(args.warmup - 1):
        run_request(model, inputs, decode_masks, args.device)

    samples = []
    for _ in range(args.runs):
        metrics, tokens, cache_lengths = run_request(model, inputs, decode_masks, args.device)
        if tokens != expected_tokens:
            raise RuntimeError("Greedy token IDs changed between requests.")
        samples.append(metrics)

    result = {
        "model": args.model,
        "revision": revision,
        "device": args.device,
        "dtype": str(model.dtype),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        **device_metadata(args.device),
        "prompt": args.prompt,
        "input_tokens": inputs["input_ids"].shape[-1],
        "warmup": args.warmup,
        "runs": args.runs,
        "decode_steps": args.decode_steps,
        "cache_lengths": cache_lengths,
        "generated_token_ids": expected_tokens,
        "generated_text": tokenizer.decode(expected_tokens, skip_special_tokens=True),
        "metrics": {name: summarize([sample[name] for sample in samples]) for name in samples[0]},
        "samples": samples,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
