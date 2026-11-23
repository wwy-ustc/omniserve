"""Replay the same audio bursts against a running OmniServe instance."""

import argparse
import hashlib
import json
import mimetypes
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Audio:
    path: Path
    body: bytes
    media_type: str

    @classmethod
    def read(cls, path):
        body = path.read_bytes()
        if not body:
            raise ValueError(f"Audio file is empty: {path}")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        # Some desktop MIME tables classify MediaRecorder WebM as video/webm.
        if path.suffix.lower() == ".webm":
            media_type = "audio/webm"
        if not media_type.startswith("audio/"):
            media_type = "application/octet-stream"
        return cls(path, body, media_type)

    def description(self):
        return {
            "file": str(self.path),
            "bytes": len(self.body),
            "sha256": hashlib.sha256(self.body).hexdigest(),
        }


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("use a positive integer")
    return number


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("use a nonnegative integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+", help="files replayed in argument order")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--mode", choices=("sequential", "pipeline", "continuous", "budget"),
                        help="fail if the running server reports a different mode")
    parser.add_argument("--concurrency", type=positive_integer, default=4,
                        help="number of concurrent uploads in each burst")
    parser.add_argument("--arrival-gap-ms", type=nonnegative_integer, default=0,
                        help="requested interval between uploads in a burst")
    parser.add_argument("--bursts", type=positive_integer, default=5,
                        help="number of measured bursts")
    parser.add_argument("--warmup", type=nonnegative_integer, default=1,
                        help="number of unmeasured bursts")
    parser.add_argument("--timeout", type=positive_integer, default=180,
                        help="response deadline in seconds per burst")
    parser.add_argument("--poll-ms", type=positive_integer, default=100)
    parser.add_argument("--output", type=Path, help="also save the JSON result here")
    args = parser.parse_args()
    if args.concurrency > 256:
        parser.error("--concurrency must be at most 256 (the server's job history limit)")
    if (args.concurrency - 1) * args.arrival_gap_ms >= args.timeout * 1000:
        parser.error("the last scheduled upload must be before --timeout")
    if urlsplit(args.url).scheme not in ("http", "https") or not urlsplit(args.url).netloc:
        parser.error("--url must be an HTTP(S) server URL")
    args.url = args.url.rstrip("/")
    return args


def request_json(base_url, endpoint, *, audio=None, timeout=15):
    headers = {"Accept": "application/json"}
    body = None
    if audio is not None:
        body = audio.body
        headers["Content-Type"] = audio.media_type
    request = Request(base_url + endpoint, data=body, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(f"{request.get_method()} {endpoint}: HTTP {exc.code} {detail}") from exc


class DecodeCollector:
    """Read each forward once and reject a result if the bounded server trace overflows."""

    def __init__(self, base_url):
        self.base_url = base_url
        snapshot = request_json(base_url, "/api/decode-trace")
        if "next_cursor" not in snapshot:
            raise RuntimeError("Update voice_server.py before collecting continuous decode traces.")
        self.cursor = snapshot["next_cursor"]
        self.server_id = snapshot.get("server_id")
        self.steps = []

    def poll(self, timeout=15):
        snapshot = request_json(
            self.base_url, f"/api/decode-trace?after={self.cursor}", timeout=timeout,
        )
        if snapshot.get("truncated"):
            raise RuntimeError("Decode trace overflowed; reduce --poll-ms and repeat the benchmark.")
        if self.server_id is not None and snapshot.get("server_id") != self.server_id:
            raise RuntimeError("Server restarted during decode trace collection.")
        if snapshot["next_cursor"] < self.cursor:
            raise RuntimeError("Server restarted during decode trace collection.")
        self.steps.extend(snapshot["decodes"])
        self.cursor = snapshot["next_cursor"]

    def for_jobs(self, jobs):
        ids = {job["id"] for job in jobs}
        return [step for step in self.steps if ids.intersection(step["job_ids"])]


def submit_audio(audio, base_url, deadline, start, start_at, arrival_index, gap_ms):
    start.wait()
    scheduled = start_at + arrival_index * gap_ms / 1000
    if scheduled >= deadline:
        raise TimeoutError("Scheduled upload is outside the burst deadline.")
    delay = scheduled - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    timeout = deadline - time.monotonic()
    if timeout <= 0:
        raise TimeoutError("Upload time exceeded the burst deadline.")
    query = urlencode({"name": audio.path.name})
    return request_json(base_url, f"/api/jobs?{query}", audio=audio, timeout=timeout)


def run_burst(executor, args, audio_files, trace_collector=None):
    # All upload threads wait for the same signal before issuing requests.
    start = threading.Event()
    start_at = time.monotonic()
    deadline = start_at + args.timeout
    futures = [
        executor.submit(submit_audio, audio, args.url, deadline, start, start_at, index,
                        args.arrival_gap_ms)
        for index, audio in enumerate(audio_files)
    ]
    start.set()
    # Collect traces during staggered uploads too: early jobs may finish before the
    # final upload is sent and otherwise disappear from the bounded trace ring.
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Upload time exceeded the burst deadline.")
        done, pending_uploads = wait(futures, timeout=min(args.poll_ms / 1000, remaining))
        for future in done:
            future.result()
        if trace_collector is not None:
            trace_collector.poll(timeout=max(0.001, min(deadline - time.monotonic(), 15)))
        if not pending_uploads:
            break
    accepted = [future.result() for future in futures]
    job_ids = [job["id"] for job in accepted]
    pending = set(job_ids)
    completed = {}
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for jobs: {', '.join(sorted(pending))}")
        listing = request_json(args.url, "/api/jobs", timeout=min(remaining, 15))
        if trace_collector is not None:
            trace_collector.poll(timeout=max(0.001, min(deadline - time.monotonic(), 15)))
        for job in listing["jobs"]:
            job_id = job["id"]
            if job_id not in pending:
                continue
            if job["status"] == "error":
                raise RuntimeError(f"{job['name']} failed: {job.get('error') or 'unknown error'}")
            if job["status"] == "done":
                completed[job_id] = job
                pending.remove(job_id)
        if pending:
            time.sleep(min(args.poll_ms / 1000, max(0, deadline - time.monotonic())))
    # Keep the input order even when jobs finish out of order.
    return [completed[job_id] for job_id in job_ids]


def percentile_summary(values):
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * 0.95
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return {
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower),
    }


def stage_overlap(jobs):
    overlaps = []
    pairs = []
    for asr in jobs:
        for llm in jobs:
            if llm["id"] == asr["id"]:
                continue
            start = max(asr["asr_started_ms"], llm["llm_started_ms"])
            end = min(asr["asr_finished_ms"], llm["finished_ms"])
            if end > start:
                overlaps.append((start, end))
                pairs.append({
                    "asr_job": asr["id"], "llm_job": llm["id"],
                    "start_ms": start, "end_ms": end, "duration_ms": end - start,
                })
    # Count wall time once when several pairs overlap at the same moment.
    union_duration = 0
    covered_until = float("-inf")
    for start, end in sorted(overlaps):
        union_duration += max(0, end - max(start, covered_until))
        covered_until = max(covered_until, end)
    return union_duration, pairs


def summarize(bursts, decode_traces):
    windows = []
    samples = {}
    stages = {
        "end_to_end_ms": ("created_ms", "finished_ms"),
        "asr_queue_ms": ("created_ms", "asr_started_ms"),
        "asr_ms": ("asr_started_ms", "asr_finished_ms"),
        "llm_queue_ms": ("asr_finished_ms", "llm_started_ms"),
        "llm_ms": ("llm_started_ms", "finished_ms"),
        # The service's historic first_token_ms field timestamps first visible text.
        "accepted_to_first_token_ms": ("created_ms", "first_token_ms"),
        "asr_to_first_visible_ms": ("asr_finished_ms", "first_token_ms"),
    }
    total_jobs = 0
    for jobs, trace in zip(bursts, decode_traces, strict=True):
        total_jobs += len(jobs)
        for job in jobs:
            for name, (start, end) in stages.items():
                if job.get(start) is not None and job.get(end) is not None:
                    samples.setdefault(name, []).append(job[end] - job[start])
        makespan = max(job["finished_ms"] for job in jobs) - min(
            job["created_ms"] for job in jobs
        )
        overlap_ms, pairs = stage_overlap(jobs)
        windows.append({
            "makespan_ms": makespan,
            "asr_llm_overlap_ms": overlap_ms,
            "overlapping_pairs": pairs,
            "jobs": jobs,
            "decodes": trace,
        })
    total_ms = sum(window["makespan_ms"] for window in windows)
    overlap_ms = sum(window["asr_llm_overlap_ms"] for window in windows)
    return {
        "requests": total_jobs,
        "shared_decode_forwards_observed": sum(
            len(step["job_ids"]) > 1 for trace in decode_traces for step in trace
        ),
        "max_decode_batch_observed": max(
            (len(step["job_ids"]) for trace in decode_traces for step in trace), default=None
        ),
        "completed_requests_per_second": 1000 * total_jobs / total_ms if total_ms else None,
        "asr_llm_overlap_ms": overlap_ms,
        "asr_llm_overlap_fraction": overlap_ms / total_ms if total_ms else None,
        "latency": {name: percentile_summary(values) for name, values in samples.items()},
        "bursts": windows,
    }


def main():
    args = parse_args()
    try:
        audio_files = [Audio.read(path) for path in args.audio]
        health_before = request_json(args.url, "/health")
        if args.mode is not None and health_before.get("mode") != args.mode:
            raise ValueError(
                f"Expected server mode {args.mode}, got {health_before.get('mode')}"
            )
        active_limit = health_before.get("max_active_jobs")
        if active_limit is not None and args.concurrency > active_limit:
            raise ValueError(f"--concurrency exceeds the server's {active_limit} active job slots")
        bursts = []
        decode_traces = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for index in range(args.warmup + args.bursts):
                is_warmup = index < args.warmup
                offset = (index if is_warmup else index - args.warmup) * args.concurrency
                cohort = [
                    audio_files[(offset + i) % len(audio_files)]
                    for i in range(args.concurrency)
                ]
                print(f"{'Warmup' if is_warmup else 'Measure'} burst {index + 1}",
                      file=sys.stderr)
                collector = DecodeCollector(args.url) if health_before["mode"] == "continuous" else None
                samples = run_burst(executor, args, cohort, collector)
                if not is_warmup:
                    bursts.append(samples)
                    trace = collector.for_jobs(samples) if collector is not None else []
                    decode_traces.append(trace)
        result = {
            "mode": health_before.get("mode"),
            "device": health_before.get("device"),
            "concurrency": args.concurrency,
            "arrival_gap_ms": args.arrival_gap_ms,
            "warmup_bursts": args.warmup,
            "audio": [audio.description() for audio in audio_files],
            "health_before": health_before,
            "health_after": request_json(args.url, "/health"),
            "decode_trace_complete": True if health_before["mode"] == "continuous" else None,
            **summarize(bursts, decode_traces),
        }
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(f"Voice benchmark failed: {exc}") from exc


if __name__ == "__main__":
    main()
