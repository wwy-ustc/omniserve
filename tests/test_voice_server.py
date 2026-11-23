"""Service contracts and worker recovery without downloading model weights."""

import hashlib
import http.client
import json
import time
import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from threading import Event, Thread
from unittest.mock import patch
from urllib.parse import urlencode

from voice_server import MAX_ACTIVE, MAX_UPLOAD_BYTES, QueueFull, VoicePipeline, make_handler


class Recognizer:
    def transcribe(self, audio):
        if audio == b"bad-asr":
            raise ValueError("invalid test audio")
        return audio.decode()


class LanguageModel:
    device = "cpu"
    dtype = "float32"
    model_id = "test-model"
    revision = "test-revision"
    max_new_tokens = 3

    def peak_cuda_bytes(self):
        return None

    def generate(self, prompt, on_text, *, enforce_budget, on_complete):
        if prompt == "bad-llm":
            raise RuntimeError("test model failure")
        on_text("reply: ")
        on_text("reply: " + prompt)
        on_complete([2, 3, 4])
        return "reply: " + prompt


def finished(pipeline, job_ids):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        jobs = [pipeline.get(job_id) for job_id in job_ids]
        if all(job["status"] in {"done", "error"} for job in jobs):
            return jobs
        time.sleep(0.005)
    raise AssertionError("workers did not finish the submitted jobs")


class WorkerTests(unittest.TestCase):
    def test_failures_do_not_stop_workers(self):
        for mode in ("sequential", "pipeline", "budget"):
            with self.subTest(mode=mode):
                pipeline = VoicePipeline(Recognizer(), LanguageModel(), mode)
                ids = [pipeline.submit(name, name.encode())["id"]
                       for name in ("bad-asr", "bad-llm", "good")]
                jobs = finished(pipeline, ids)
                self.assertEqual([job["status"] for job in jobs], ["error", "error", "done"])
                self.assertEqual(jobs[0]["error"], "invalid test audio")
                self.assertEqual(jobs[2]["answer"], "reply: good")
                self.assertEqual(jobs[2]["generated_tokens"], 3)
                self.assertEqual(jobs[2]["token_sha256"], hashlib.sha256(b"[2,3,4]").hexdigest())
                self.assertLessEqual(jobs[2]["created_ms"], jobs[2]["first_token_ms"])
                self.assertLessEqual(jobs[2]["first_token_ms"], jobs[2]["finished_ms"])

    def test_pipeline_overlaps_asr_with_llm(self):
        generation_started = Event()
        second_asr_started = Event()

        class OverlapASR(Recognizer):
            def transcribe(self, audio):
                if audio == b"second":
                    if not generation_started.wait(2):
                        raise AssertionError("first generation never started")
                    second_asr_started.set()
                return super().transcribe(audio)

        class OverlapLLM(LanguageModel):
            def generate(self, prompt, on_text, **kwargs):
                if prompt == "first":
                    generation_started.set()
                    if not second_asr_started.wait(2):
                        raise AssertionError("ASR did not overlap generation")
                return super().generate(prompt, on_text, **kwargs)

        pipeline = VoicePipeline(OverlapASR(), OverlapLLM(), "pipeline")
        ids = [pipeline.submit(name, name.encode())["id"] for name in ("first", "second")]
        self.assertEqual([job["status"] for job in finished(pipeline, ids)], ["done", "done"])
        self.assertTrue(second_asr_started.is_set())

    def test_active_limit_is_atomic_for_concurrent_submissions(self):
        release = Event()

        class BlockedASR(Recognizer):
            def transcribe(self, audio):
                release.wait(3)
                return super().transcribe(audio)

        pipeline = VoicePipeline(BlockedASR(), LanguageModel(), "pipeline")

        def submit(_):
            try:
                return pipeline.submit("test", b"good")["id"]
            except QueueFull:
                return None

        try:
            with ThreadPoolExecutor(max_workers=12) as executor:
                results = list(executor.map(submit, range(MAX_ACTIVE + 8)))
            self.assertEqual(sum(job_id is not None for job_id in results), MAX_ACTIVE)
            self.assertEqual(len(pipeline.list_jobs()), MAX_ACTIVE)
        finally:
            release.set()
        self.assertTrue(all(job["status"] == "done"
                            for job in finished(pipeline, [item for item in results if item])))

    def test_history_discards_finished_jobs(self):
        with patch("voice_server.MAX_HISTORY", 3):
            pipeline = VoicePipeline(Recognizer(), LanguageModel(), "sequential")
            ids = []
            for _ in range(5):
                job_id = pipeline.submit("test", b"good")["id"]
                ids.append(job_id)
                finished(pipeline, [job_id])
            self.assertEqual([job["id"] for job in pipeline.list_jobs()], ids[-3:][::-1])
            self.assertIsNone(pipeline.get(ids[0]))

    def test_trace_cursors_detect_overflow(self):
        pipeline = VoicePipeline(Recognizer(), LanguageModel(), "pipeline")
        pipeline.decode_trace = deque(maxlen=2)
        for index in range(3):
            pipeline.record_decode([f"job{index}"], [index + 10])
        stale = pipeline.decode_snapshot(after=0)
        self.assertTrue(stale["truncated"])
        self.assertEqual(stale["next_cursor"], 3)
        snapshot = pipeline.decode_snapshot(after=1)
        self.assertFalse(snapshot["truncated"])
        self.assertEqual([step["sequence"] for step in snapshot["decodes"]], [2, 3])
        self.assertEqual(pipeline.decode_snapshot(after=3)["decodes"], [])


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = VoicePipeline(Recognizer(), LanguageModel(), "pipeline")
        handler = make_handler(cls.pipeline, {"/": ("text/html", b"console")}, 30, 3)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            data = response.read()
            return response.status, (json.loads(data) if "json" in response.getheader("Content-Type", "") else data)
        finally:
            connection.close()

    def test_submit_and_poll_roundtrip(self):
        status, accepted = self.request("POST", "/api/jobs?" + urlencode({"name": "中文.wav"}), b"hello")
        self.assertEqual(status, 202)
        finished(self.pipeline, [accepted["id"]])
        status, job = self.request("GET", "/api/jobs/" + accepted["id"])
        self.assertEqual(status, 200)
        self.assertEqual(job["name"], "中文.wav")
        self.assertEqual(job["transcript"], "hello")
        self.assertEqual(job["answer"], "reply: hello")

    def test_origin_host_and_upload_bounds(self):
        cases = [
            ("GET", "/api/jobs", None, {"Host": "evil.invalid"}, 403),
            ("POST", "/api/jobs", b"x", {"Origin": "https://evil.invalid"}, 403),
            ("POST", "/api/jobs", b"", {}, 400),
            ("POST", "/api/jobs", b"x", {"Content-Length": "invalid"}, 400),
            ("POST", "/api/jobs", b"x", {"Content-Length": str(MAX_UPLOAD_BYTES + 1)}, 413),
            ("GET", "/api/jobs/missing", None, {}, 404),
            ("GET", "/api/decode-trace?after=-1", None, {}, 400),
            ("GET", "/api/decode-trace?after=invalid", None, {}, 400),
        ]
        for method, path, body, headers, expected in cases:
            with self.subTest(method=method, path=path, headers=headers):
                self.assertEqual(self.request(method, path, body, headers)[0], expected)

    def test_health_and_trace_contract(self):
        status, health = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["max_active_jobs"], MAX_ACTIVE)
        self.assertEqual(health["decode_trace_capacity"], 2048)
        status, trace = self.request("GET", "/api/decode-trace?after=0")
        self.assertEqual(status, 200)
        self.assertEqual(trace["decodes"], [])
        self.assertEqual(trace["next_cursor"], 0)
        self.assertFalse(trace["truncated"])


if __name__ == "__main__":
    unittest.main()
