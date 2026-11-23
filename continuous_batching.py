"""Run real shared Qwen decode forwards with a tick-based arrival schedule."""

import argparse
import json
from collections import deque
from time import perf_counter

import torch
import transformers

from batching_baseline import (
    PROMPTS, benchmark, encode, eos_ids, generate, load_local, prepare, trim,
)
from workers.decode_scheduler import Request, Scheduler
from workers.llm_worker import (
    DEVICE_CHOICES, MODEL_ID, device_metadata, peak_memory, positive_int,
    reset_peak_memory, synchronize,
)


@torch.inference_mode()
def run(scheduler, requests):
    pending = deque(sorted(enumerate(requests), key=lambda entry: entry[1].arrival))
    active = {}
    completed = {}
    trace = []
    tick = 0
    while pending or active:
        if not active:
            tick = max(tick, pending[0][1].arrival)
        # Newly admitted requests emit their first token in prefill and join
        # an existing decode group on the following tick.
        decode_ids = list(active)
        admitted = []
        while pending and pending[0][1].arrival <= tick and len(active) < scheduler.max_active:
            request_id, request = pending.popleft()
            active[request_id] = scheduler.start(request)
            admitted.append(request_id)
        lengths = scheduler.decode([active[i] for i in decode_ids]) if decode_ids else []
        finished = [i for i, state in active.items() if scheduler.finished(state)]
        for i in finished:
            completed[i] = active.pop(i).generated
        trace.append({
            "tick": tick, "admitted": admitted, "decode_request_ids": decode_ids,
            "decode_cache_lengths": lengths, "finished": finished,
        })
        tick += 1
    return [completed[i] for i in range(len(requests))], trace


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--compare", action="store_true",
                        help="Benchmark all three modes on the same simultaneous requests.")
    parser.add_argument("--strict-token-match", action="store_true",
                        help="Fail if batch shapes produce different greedy token IDs.")
    parser.add_argument("--runs", type=positive_int, default=6, help="Measured trials per mode in --compare.")
    parser.add_argument("--max-new-tokens", type=positive_int, default=32, help="Token limit in --compare.")
    args = parser.parse_args()
    model, tokenizer, revision = load_local(args)
    if args.compare:
        rows = encode(tokenizer, PROMPTS)
        requests = [Request(0, row, args.max_new_tokens) for row in rows]
        scheduler = Scheduler(model, args.device, max_active=len(requests))
        result = benchmark(model, tokenizer, rows, args, continuous=lambda: run(scheduler, requests))
        if args.strict_token_match and not all(
            result["matching_token_ids"] + result["continuous_matching_token_ids"]
        ):
            raise RuntimeError("The three modes produced different token IDs on this workload.")
        print(json.dumps({
            "model": args.model, "revision": revision, "device": args.device,
            "dtype": str(model.dtype), "torch": torch.__version__,
            "transformers": transformers.__version__, "warmup_per_mode": args.warmup,
            **device_metadata(args.device),
            "runs_per_mode": args.runs, "max_new_tokens": args.max_new_tokens,
            "input_lengths": list(map(len, rows)), "prompts": PROMPTS,
            "arrival_ticks": [r.arrival for r in requests], **result,
        }, ensure_ascii=False, indent=2))
        return

    rows = encode(tokenizer, (PROMPTS[2], PROMPTS[0], PROMPTS[1]))
    requests = [Request(0, rows[0], 3), Request(0, rows[1], 32), Request(2, rows[2], 8)]
    scheduler = Scheduler(model, args.device, max_active=2)

    for _ in range(args.warmup):
        run(scheduler, requests)
    synchronize(args.device)
    reset_peak_memory(args.device)
    start = perf_counter()
    tokens, trace = run(scheduler, requests)
    synchronize(args.device)
    elapsed = perf_counter() - start
    memory = peak_memory(args.device)

    stops = eos_ids(model)
    expected = [
        trim(generate(model, prepare(tokenizer, [r.input_ids], args.device),
                      r.max_new_tokens, tokenizer.pad_token_id)[0].tolist(), stops)
        for r in requests
    ]
    if args.strict_token_match and tokens != expected:
        raise RuntimeError(f"Continuous outputs differ from standalone generation: {tokens} != {expected}")
    late_admission = any(2 in step["admitted"] and step["decode_request_ids"] == [1] for step in trace)
    shared_decode = any(step["decode_request_ids"] == [1, 2]
                        and len(set(step["decode_cache_lengths"])) == 2 for step in trace)
    if not (late_admission and shared_decode):
        raise RuntimeError(f"Late admission or shared unequal-length decode was missing: {trace}")

    print(json.dumps({
        "model": args.model, "revision": revision, "device": args.device,
        "dtype": str(model.dtype), "max_active": scheduler.max_active, "seconds": elapsed,
        **device_metadata(args.device), **memory,
        "matches_standalone": tokens == expected,
        "shared_decode_forwards": sum(len(step["decode_request_ids"]) > 1 for step in trace),
        "requests": [
            {"arrival": r.arrival, "input_tokens": len(r.input_ids),
             "max_new_tokens": r.max_new_tokens, "token_ids": output,
             "finish": "eos" if output[-1] in stops else "length",
             "text": tokenizer.decode(output, skip_special_tokens=True),
             "matches_standalone": output == reference,
             "standalone_token_ids": reference}
            for r, output, reference in zip(requests, tokens, expected, strict=True)
        ],
        "trace": trace,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
