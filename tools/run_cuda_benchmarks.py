"""Save real Qwen CUDA benchmarks as JSON, with command logs and failure status."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--strict-token-match", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.max_new_tokens < 1:
        parser.error("runs and max-new-tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = ["--device", "cuda"] + (["--local-files-only"] if args.local_files_only else [])
    commands = [
        ("single-request", ["workers/llm_worker.py", *common, "--warmup", "1",
                            "--runs", str(args.runs), "--decode-steps", str(args.max_new_tokens)]),
        ("batch-comparison", ["continuous_batching.py", *common, "--compare", "--warmup", "1",
                              "--runs", str(args.runs), "--max-new-tokens", str(args.max_new_tokens)]),
        ("arrival-topology", ["continuous_batching.py", *common, "--warmup", "1"]),
    ]
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "runs": []}
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    for name, command in commands:
        if args.strict_token_match and name != "single-request":
            command.append("--strict-token-match")
        completed = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", env=environment)
        (args.output_dir / f"{name}.log").write_text(completed.stderr, encoding="utf-8")
        record = {"name": name, "command": command, "returncode": completed.returncode}
        try:
            if completed.returncode:
                raise ValueError("Benchmark failed; see the corresponding .log file.")
            result = json.loads(completed.stdout)
            (args.output_dir / f"{name}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            record["status"] = "passed"
        except ValueError as exc:
            record.update(status="failed", error=str(exc))
            (args.output_dir / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
            report["status"] = "failed"
        report["runs"].append(record)
        print(f"{name}: {record['status']}", flush=True)
        if record["status"] == "failed":
            break
    (args.output_dir / "run-summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
