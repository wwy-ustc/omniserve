"""Validate CUDA and real shared decode using tiny random Qwen weights, offline."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate(report):
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Install CUDA-enabled PyTorch and requirements-voice.txt first.") from exc
    report.update(torch=torch.__version__, transformers=transformers.__version__,
                  cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: check the PyTorch CUDA wheel and NVIDIA driver.")

    from transformers import Qwen2Config, Qwen2ForCausalLM
    from continuous_batching import run
    from memory_budget import cuda_admission_budget
    from voice_llm import VoiceLanguageModel
    from workers.decode_scheduler import Request, Scheduler
    from workers.llm_worker import device_metadata, peak_memory, synchronize

    report.update(device_metadata("cuda"))
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    # This exercises CUDA kernels but needs no network or model download.
    tensor = torch.randn(32, 32, device="cuda", dtype=torch.float16)
    if not torch.isfinite(tensor @ tensor.T).all().item():
        raise RuntimeError("CUDA FP16 matrix multiply produced non-finite values.")
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128,
                         bos_token_id=1, pad_token_id=0, eos_token_id=None)
    model = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.float16).eval()
    scheduler = Scheduler(model, "cuda", max_active=2)
    requests = [Request(0, [1, 3, 5], 3), Request(0, [1, 7, 8, 9, 10], 9),
                Request(2, [1, 11, 12, 13], 4)]
    synchronize("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    with torch.inference_mode():
        tokens, trace = run(scheduler, requests)
        synchronize("cuda")
        elapsed = perf_counter() - started
        memory = peak_memory("cuda")
        reference = []
        for request in requests:
            state = scheduler.start(request)
            while not scheduler.finished(state):
                scheduler.decode([state])
            if state.cache is not None:
                raise RuntimeError("A completed request retained its KV cache.")
            reference.append(state.generated)
    if list(map(len, tokens)) != [3, 9, 4]:
        raise RuntimeError("A request did not respect its output token limit.")
    if not any(len(step["decode_request_ids"]) > 1
               and len(set(step["decode_cache_lengths"])) > 1 for step in trace):
        raise RuntimeError("No shared decode with unequal cache lengths was observed.")
    if not any(2 in step["admitted"] and step["decode_request_ids"] == [1] for step in trace):
        raise RuntimeError("A late request did not reuse a completed request's slot.")
    llm = VoiceLanguageModel.__new__(VoiceLanguageModel)
    llm.model, llm.device, llm.dtype, llm.max_new_tokens = model, "cuda", torch.float16, 9
    estimate = llm._budget(5)
    rejection = cuda_admission_budget(kv_bytes=1024, input_tokens=5, attention_heads=4,
                                      driver_free_bytes=0, total_bytes=4 * 2**30,
                                      reserved_bytes=0, allocated_bytes=0)
    if rejection["cuda_admitted"]:
        raise RuntimeError("Memory estimator accepted a zero-memory budget.")
    report.update(
        validation_scope="tiny random Qwen FP16; not pretrained-model or voice performance",
        seconds=elapsed, **memory, budget=estimate,
        zero_memory_rejected=True, token_limits_respected=True, cache_released=True,
        matching_token_ids=[a == b for a, b in zip(tokens, reference, strict=True)],
        shared_decode_forwards=sum(len(step["decode_request_ids"]) > 1 for step in trace),
        generated_token_ids=tokens, standalone_token_ids=reference, trace=trace,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cuda-validation.json")
    parser.add_argument("--strict-token-match", action="store_true")
    args = parser.parse_args()
    report = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.platform(),
              "status": "failed"}
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True,
                                timeout=10, check=False)
        report["nvidia_smi"] = driver.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi_error"] = str(exc)
    try:
        validate(report)
        if args.strict_token_match and not all(report["matching_token_ids"]):
            raise RuntimeError("FP16 shared and standalone token IDs differ; inspect the report.")
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
