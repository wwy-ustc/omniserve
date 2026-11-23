"""Offline three-request greedy generation benchmark for OmniServe."""

import argparse
import json
from statistics import median
from time import perf_counter

import torch
import transformers

from workers.llm_worker import MODEL_ID, load_model, positive_int, synchronize


PROMPTS = (
    "Reply with just the word hello.",
    "In one short sentence, explain what a GPU does.",
    "Explain why serving multiple language model requests in one batch can "
    "improve throughput. Include an example with three requests of different lengths.",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("mps", "cpu"), default=(
        "mps" if torch.backends.mps.is_available() else "cpu"
    ))
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--max-new-tokens", type=positive_int, default=32)
    return parser.parse_args()


def encode(tokenizer, prompts):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
        for prompt in prompts
    ]


def prepare(tokenizer, rows, device):
    return tokenizer.pad(
        {"input_ids": rows}, padding=True, return_attention_mask=True, return_tensors="pt",
    ).to(device)


def eos_ids(model):
    eos = model.generation_config.eos_token_id
    if eos is None:
        return set()
    return {eos} if isinstance(eos, int) else set(eos)


def trim(row, stops):
    for index, token in enumerate(row):
        if token in stops:
            return row[:index + 1]
    return row


def generate(model, inputs, limit, pad):
    output = model.generate(
        **inputs, max_new_tokens=limit, do_sample=False, num_beams=1,
        use_cache=True, pad_token_id=pad, repetition_penalty=1.0,
    )
    return output[:, inputs["input_ids"].shape[-1]:]


def benchmark(model, tokenizer, rows, args, *, continuous=None):
    single = [prepare(tokenizer, [row], args.device) for row in rows]
    batch = prepare(tokenizer, rows, args.device)
    pad = tokenizer.pad_token_id
    stops = eos_ids(model)

    def timed(mode):
        synchronize(args.device)
        start = perf_counter()
        if mode == "continuous_batch":
            tokens, trace = continuous()
        elif mode == "static_batch":
            outputs = [generate(model, batch, args.max_new_tokens, pad)]
        else:
            outputs = [generate(model, inputs, args.max_new_tokens, pad) for inputs in single]
        if mode != "continuous_batch":
            tokens = [trim(row, stops) for output in outputs for row in output.tolist()]
        synchronize(args.device)
        elapsed = perf_counter() - start
        sample = {
            "seconds": elapsed,
            "generated_tokens": sum(map(len, tokens)),
            "tokens_per_second": sum(map(len, tokens)) / elapsed,
            "completed_requests_per_second": len(tokens) / elapsed,
        }
        if mode == "continuous_batch":
            sample["shared_decode_forwards"] = sum(
                len(step["decode_request_ids"]) > 1 for step in trace
            )
        return sample, tokens

    modes = ("sequential", "static_batch")
    if continuous is not None:
        modes += ("continuous_batch",)
    samples = {mode: [] for mode in modes}
    reference = {}
    for trial in range(args.warmup + args.runs):
        offset = trial % len(modes)
        order = modes[offset:] + modes[:offset]
        for mode in order:
            sample, tokens = timed(mode)
            if trial >= args.warmup:
                if reference.setdefault(mode, tokens) != tokens:
                    raise RuntimeError(f"Greedy outputs changed across {mode} trials")
                samples[mode].append(sample)

    result = {
        "padded_input_width": batch["input_ids"].shape[1],
        "matching_token_ids": [
            a == b for a, b in zip(reference["sequential"], reference["static_batch"], strict=True)
        ],
        "modes": {
            mode: {
                "median": {key: median(s[key] for s in samples[mode]) for key in samples[mode][0]},
                "outputs": [
                    {
                        "token_ids": row,
                        "text": tokenizer.decode(row, skip_special_tokens=True),
                        "finish": "eos" if row[-1] in stops else "length",
                    }
                    for row in reference[mode]
                ],
                "samples": samples[mode],
            }
            for mode in modes
        },
    }

    if continuous is not None:
        result["continuous_matching_token_ids"] = [
            a == b for a, b in zip(reference["sequential"], reference["continuous_batch"], strict=True)
        ]
    return result


def load_local(args):
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable. Use --device cpu for a CPU run.")
    args.local_files_only = True
    model, tokenizer, revision = load_model(args)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("The tokenizer must define a PAD or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, revision


@torch.inference_mode()
def main():
    args = parse_args()
    model, tokenizer, revision = load_local(args)
    rows = encode(tokenizer, PROMPTS)
    result = benchmark(model, tokenizer, rows, args)
    print(json.dumps({
        "model": args.model, "revision": revision, "device": args.device,
        "dtype": str(model.dtype), "torch": torch.__version__,
        "transformers": transformers.__version__, "warmup_per_mode": args.warmup,
        "runs_per_mode": args.runs, "max_new_tokens": args.max_new_tokens,
        "input_lengths": list(map(len, rows)), "eos_ids": sorted(eos_ids(model)),
        "pad_id": tokenizer.pad_token_id, "prompts": PROMPTS, **result,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
