"""Run the local OmniServe voice console with python voice_server.py."""

import argparse
import json
import logging
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from time import monotonic_ns, time_ns
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_ACTIVE = 24
MAX_HISTORY = 256
TERMINAL = {"done", "error"}
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
LOGGER = logging.getLogger("omniserve")
CLOCK_OFFSET_NS = time_ns() - monotonic_ns()


def now_ms():
    return (monotonic_ns() + CLOCK_OFFSET_NS) // 1_000_000


@dataclass
class Job:
    id: str
    name: str
    created_ms: int
    status: str = "queued"
    transcript: str = ""
    answer: str = ""
    error: str = ""
    asr_started_ms: int | None = None
    asr_finished_ms: int | None = None
    llm_started_ms: int | None = None
    first_token_ms: int | None = None
    finished_ms: int | None = None
    generated_tokens: int | None = None
    token_sha256: str | None = None


class QueueFull(Exception):
    pass


class VoicePipeline:
    def __init__(self, recognizer, llm, mode, max_batch_size=4):
        self.recognizer = recognizer
        self.llm = llm
        self.mode = mode
        self.scheduler = None
        if mode == "continuous":
            from workers.decode_scheduler import Scheduler

            self.scheduler = Scheduler(llm.model, llm.device, max_batch_size)
        self.decode_trace = deque(maxlen=2048)
        self.decode_count = 0
        self.server_id = uuid4().hex
        self.jobs = OrderedDict()
        self.lock = Lock()
        self.asr_queue = Queue()
        self.llm_queue = Queue()
        Thread(target=self.run_asr, name="voice-asr", daemon=True).start()
        if self.mode != "sequential":
            worker = self.run_continuous if self.scheduler is not None else self.run_llm
            Thread(target=worker, name="voice-llm", daemon=True).start()

    def submit(self, name, audio):
        with self.lock:
            if sum(job.status not in TERMINAL for job in self.jobs.values()) >= MAX_ACTIVE:
                raise QueueFull
            while len(self.jobs) >= MAX_HISTORY:
                oldest_finished = next(
                    job_id for job_id, job in self.jobs.items() if job.status in TERMINAL
                )
                del self.jobs[oldest_finished]
            job = Job(uuid4().hex, name, now_ms())
            self.jobs[job.id] = job
            self.asr_queue.put((job, audio))
            return asdict(job)

    def list_jobs(self):
        with self.lock:
            return [asdict(job) for job in reversed(self.jobs.values())]

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return asdict(job) if job is not None else None

    def update(self, job, **fields):
        with self.lock:
            for name, value in fields.items():
                setattr(job, name, value)

    def fail(self, job, exc):
        LOGGER.warning("Request %s failed: %s", job.id, exc)
        if isinstance(exc, ValueError):
            message = str(exc)
        elif isinstance(exc, RuntimeError) and self.mode == "budget":
            message = str(exc)
        else:
            message = "处理请求时发生错误，请检查服务日志。"
        self.update(job, status="error", error=message, finished_ms=now_ms())

    def run_asr(self):
        while True:
            job, audio = self.asr_queue.get()
            try:
                self.update(job, status="transcribing", asr_started_ms=now_ms())
                transcript = self.recognizer.transcribe(audio)
                self.update(job, status="waiting", transcript=transcript, asr_finished_ms=now_ms())
                if self.mode == "sequential":
                    self.generate(job)
                else:
                    self.llm_queue.put(job)
            except Exception as exc:
                self.fail(job, exc)
            finally:
                del audio
                self.asr_queue.task_done()

    def run_llm(self):
        while True:
            job = self.llm_queue.get()
            try:
                self.generate(job)
            finally:
                self.llm_queue.task_done()

    def recent_decodes(self):
        with self.lock:
            return list(self.decode_trace)

    def record_decode(self, job_ids, cache_lengths):
        with self.lock:
            self.decode_count += 1
            self.decode_trace.append({
                "sequence": self.decode_count, "at_ms": now_ms(),
                "job_ids": list(job_ids), "cache_lengths": list(cache_lengths),
            })

    def decode_snapshot(self, after=None):
        with self.lock:
            oldest = self.decode_trace[0]["sequence"] if self.decode_trace else self.decode_count + 1
            return {
                "server_id": self.server_id,
                "decodes": [step for step in self.decode_trace
                            if after is None or step["sequence"] > after],
                "next_cursor": self.decode_count,
                "oldest_cursor": oldest,
                "truncated": after is not None and after < oldest - 1,
            }

    def publish(self, job, text):
        with self.lock:
            if text and job.first_token_ms is None:
                job.first_token_ms = now_ms()
            job.answer = text

    def complete(self, job, answer, tokens):
        digest = sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        self.update(job, status="done", answer=answer, finished_ms=now_ms(),
                    generated_tokens=len(tokens), token_sha256=digest)

    def generate(self, job):
        self.update(job, status="generating", llm_started_ms=now_ms())
        tokens = []
        try:
            answer = self.llm.generate(
                job.transcript, lambda text: self.publish(job, text),
                enforce_budget=self.mode == "budget", on_complete=tokens.extend,
            )
            self.complete(job, answer, tokens)
        except Exception as exc:
            self.fail(job, exc)

    def run_continuous(self):
        from workers.decode_scheduler import Request

        active = {}
        while True:
            ready = list(active)
            # Wait only when idle. Admit at most one queue entry per free slot per round.
            for _ in range(self.scheduler.max_active - len(ready)):
                try:
                    job = self.llm_queue.get(block=not active)
                except Empty:
                    break
                self.update(job, status="generating", llm_started_ms=now_ms())
                try:
                    request = Request(0, self.llm.prompt_token_ids(job.transcript), self.llm.max_new_tokens)
                    state = self.scheduler.start(request)
                    self.publish_continuous(job, state)
                    if not self.scheduler.finished(state):
                        active[job.id] = (job, state)
                        continue
                except Exception as exc:
                    self.fail(job, exc)
                self.llm_queue.task_done()

            if not ready:
                continue
            try:
                lengths = self.scheduler.decode([active[job_id][1] for job_id in ready])
                self.record_decode(ready, lengths)
            except Exception as exc:
                # A failed forward may have mutated cache state, so fail its whole group.
                for job_id in ready:
                    job, _ = active.pop(job_id)
                    self.fail(job, exc)
                    self.llm_queue.task_done()
                continue
            for job_id in ready:
                job, state = active[job_id]
                try:
                    self.publish_continuous(job, state)
                    if not self.scheduler.finished(state):
                        continue
                except Exception as exc:
                    self.fail(job, exc)
                del active[job_id]
                self.llm_queue.task_done()

    def publish_continuous(self, job, state):
        answer = self.llm.visible_text(state.generated)
        if answer != job.answer:
            self.publish(job, answer)
        if self.scheduler.finished(state):
            self.complete(job, answer, state.generated)


def make_handler(pipeline, assets, max_audio_seconds, max_new_tokens):
    class VoiceHandler(BaseHTTPRequestHandler):
        def allowed_host(self):
            port = self.server.server_port
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def setup(self):
            self.request.settimeout(20)
            super().setup()

        def log_message(self, fmt, *args):
            # Polling shouldn't flood the console.
            LOGGER.debug(fmt, *args)

        def send_bytes(self, status, mime_type, content):
            try:
                self.send_response(status)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, status, data):
            self.send_bytes(
                status, "application/json; charset=utf-8",
                json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode(),
            )

        def do_GET(self):
            if not self.allowed_host():
                self.send_json(HTTPStatus.FORBIDDEN, {"error": "请使用本机地址访问 OmniServe。"})
                return
            url = urlsplit(self.path)
            path = url.path
            if path in assets:
                mime_type, content = assets[path]
                self.send_bytes(HTTPStatus.OK, mime_type, content)
            elif path == "/health":
                peak = pipeline.llm.peak_cuda_bytes()
                self.send_json(HTTPStatus.OK, {
                    "device": pipeline.llm.device,
                    "asr_device": "cpu",
                    "llm_dtype": str(pipeline.llm.dtype),
                    "max_batch_size": pipeline.scheduler.max_active if pipeline.scheduler else 1,
                    "max_active_jobs": MAX_ACTIVE,
                    "server_id": pipeline.server_id,
                    "decode_trace_capacity": pipeline.decode_trace.maxlen,
                    "mode": pipeline.mode,
                    "llm_model": pipeline.llm.model_id,
                    "llm_revision": pipeline.llm.revision,
                    "max_audio_seconds": max_audio_seconds,
                    "max_new_tokens": max_new_tokens,
                    "cuda_peak_allocated_mb": None if peak is None else peak / (1024 ** 2),
                })
            elif path == "/api/decode-trace":
                try:
                    raw = parse_qs(url.query).get("after", [None])[0]
                    after = None if raw is None else int(raw)
                    if after is not None and after < 0:
                        raise ValueError
                except ValueError:
                    self.send_json(HTTPStatus.BAD_REQUEST, {"error": "after must be a nonnegative cursor."})
                    return
                self.send_json(HTTPStatus.OK, pipeline.decode_snapshot(after))
            elif path == "/api/jobs":
                jobs = pipeline.list_jobs()
                self.send_json(HTTPStatus.OK, {
                    "now_ms": now_ms(), "device": pipeline.llm.device,
                    "mode": pipeline.mode, "jobs": jobs,
                })
            elif path.startswith("/api/jobs/"):
                job = pipeline.get(path.removeprefix("/api/jobs/"))
                if job is None:
                    self.send_json(HTTPStatus.NOT_FOUND, {"error": "找不到这条请求。"})
                else:
                    self.send_json(HTTPStatus.OK, job)
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "找不到此页面。"})

        def do_POST(self):
            url = urlsplit(self.path)
            if url.path != "/api/jobs":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "找不到此页面。"})
                return
            origin = self.headers.get("Origin")
            if not self.allowed_host() or (origin and origin != f"http://{self.headers.get('Host')}"):
                self.send_json(HTTPStatus.FORBIDDEN, {"error": "请从 OmniServe 网页提交音频。"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length < 1:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "音频内容为空。"})
                return
            if length > MAX_UPLOAD_BYTES:
                self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "单个文件最大 12 MiB。"})
                return
            name = parse_qs(url.query).get("name", ["录音"])[0][:120]
            try:
                audio = self.rfile.read(length)
            except TimeoutError:
                self.send_json(HTTPStatus.REQUEST_TIMEOUT, {"error": "上传超时，请重试。"})
                return
            if len(audio) != length:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "音频上传不完整。"})
                return
            try:
                job = pipeline.submit(name, audio)
            except QueueFull:
                self.send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "队列已满，稍后再试。"})
                return
            self.send_json(HTTPStatus.ACCEPTED, job)

    return VoiceHandler


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=positive_int, default=8765)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mode", choices=("sequential", "pipeline", "continuous", "budget"), default="pipeline")
    parser.add_argument("--max-batch-size", type=positive_int, default=4)
    parser.add_argument("--max-new-tokens", type=positive_int, default=96)
    parser.add_argument("--cpu-threads", type=positive_int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "budget" and args.device not in ("auto", "cuda"):
        parser.error("budget mode requires CUDA")
    if args.max_batch_size > MAX_ACTIVE:
        parser.error(f"max-batch-size must be at most {MAX_ACTIVE}")
    if args.port > 65535:
        parser.error("port must be at most 65535")

    import torch
    from voice_asr import MAX_AUDIO_SECONDS, VoiceRecognizer
    from voice_llm import VoiceLanguageModel

    if args.mode == "budget" and not torch.cuda.is_available():
        parser.error("budget mode requires an available CUDA device")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(args.cpu_threads)
    web = Path(__file__).resolve().parent / "web"
    assets = {route: (mime, (web / name).read_bytes()) for route, (name, mime) in STATIC_FILES.items()}
    LOGGER.info("Loading local ASR and LLM weights…")
    recognizer = VoiceRecognizer(args.local_files_only)
    llm = VoiceLanguageModel(args.device, args.local_files_only, args.max_new_tokens)
    pipeline = VoicePipeline(recognizer, llm, args.mode, args.max_batch_size)
    handler = make_handler(pipeline, assets, MAX_AUDIO_SECONDS, args.max_new_tokens)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"OmniServe · {args.mode} · ASR cpu · LLM {llm.device}", flush=True)
        print(f"Local: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
