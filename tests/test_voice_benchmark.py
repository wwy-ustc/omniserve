"""Benchmark measurement and trace integrity checks using deterministic records."""

import argparse
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from benchmarks import voice_benchmark as benchmark


def job(job_id, created=0, asr_start=0, asr_end=10, llm_start=10, finish=30):
    return {"id": job_id, "name": job_id, "status": "done", "created_ms": created,
            "asr_started_ms": asr_start, "asr_finished_ms": asr_end,
            "llm_started_ms": llm_start, "first_token_ms": llm_start + 1,
            "finished_ms": finish}


class SummaryTests(unittest.TestCase):
    def test_overlap_counts_shared_wall_time_once(self):
        jobs = [job("a", asr_end=10, llm_start=10, finish=40),
                job("b", asr_start=10, asr_end=20, llm_start=20, finish=40),
                job("c", asr_start=20, asr_end=30, llm_start=30, finish=40)]
        duration, pairs = benchmark.stage_overlap(jobs)
        self.assertEqual(duration, 20)
        self.assertEqual(sum(pair["duration_ms"] for pair in pairs), 30)
        result = benchmark.summarize([jobs], [[{"job_ids": ["a", "b"]}]])
        self.assertEqual(result["completed_requests_per_second"], 75)
        self.assertEqual(result["asr_llm_overlap_fraction"], 0.5)
        self.assertEqual(result["shared_decode_forwards_observed"], 1)

    def test_sequential_has_no_overlap(self):
        jobs = [job("a"), job("b", created=0, asr_start=30, asr_end=40,
                              llm_start=40, finish=60)]
        self.assertEqual(benchmark.stage_overlap(jobs), (0, []))

    def test_impossible_staggered_deadline_rejected(self):
        with patch("sys.argv", ["voice_benchmark.py", "one.wav", "--concurrency", "4",
                                "--arrival-gap-ms", "1000", "--timeout", "3"]):
            with self.assertRaises(SystemExit):
                benchmark.parse_args()


class CollectorTests(unittest.TestCase):
    def test_cursor_avoids_duplicates_and_filters_other_jobs(self):
        responses = [
            {"next_cursor": 5, "decodes": []},
            {"next_cursor": 7, "decodes": [{"sequence": 6, "job_ids": ["a"]},
                                            {"sequence": 7, "job_ids": ["other"]}]},
            {"next_cursor": 7, "decodes": []},
        ]
        with patch.object(benchmark, "request_json", side_effect=responses) as request:
            collector = benchmark.DecodeCollector("http://local")
            collector.poll()
            collector.poll()
        self.assertEqual([step["sequence"] for step in collector.for_jobs([job("a")])], [6])
        self.assertEqual(request.call_args_list[1].args[1], "/api/decode-trace?after=5")
        self.assertEqual(request.call_args_list[2].args[1], "/api/decode-trace?after=7")

    def test_overflow_and_restart_are_not_reported_as_complete(self):
        for response, expected in [
            ({"next_cursor": 100, "decodes": [], "truncated": True}, "overflowed"),
            ({"next_cursor": 0, "decodes": []}, "restarted"),
        ]:
            with self.subTest(response=response):
                with patch.object(benchmark, "request_json", side_effect=[
                    {"next_cursor": 5, "decodes": []}, response,
                ]):
                    collector = benchmark.DecodeCollector("http://local")
                    with self.assertRaisesRegex(RuntimeError, expected):
                        collector.poll()

    def test_burst_collects_before_all_staggered_uploads_finish(self):
        args = argparse.Namespace(url="http://local", timeout=2, poll_ms=5, arrival_gap_ms=60)
        audios = [benchmark.Audio(Path(name), name.encode(), "audio/wav") for name in ("a", "b")]
        calls = []

        class Collector:
            def poll(self, timeout):
                calls.append("trace")

        def request(_base, endpoint, **_kwargs):
            if endpoint.startswith("/api/jobs?"):
                job_id = endpoint.split("=", 1)[1]
                calls.append(job_id)
                return {"id": job_id}
            return {"jobs": [job("b"), job("a")]}

        with patch.object(benchmark, "request_json", side_effect=request):
            with ThreadPoolExecutor(max_workers=2) as executor:
                jobs = benchmark.run_burst(executor, args, audios, Collector())
        self.assertEqual([entry["id"] for entry in jobs], ["a", "b"])
        self.assertLess(calls.index("trace"), calls.index("b"))

    def test_restart_with_a_larger_cursor_still_invalidates_measurement(self):
        with patch.object(benchmark, "request_json", side_effect=[
            {"next_cursor": 5, "decodes": [], "server_id": "first-instance"},
            {"next_cursor": 20, "decodes": [], "server_id": "second-instance"},
        ]):
            collector = benchmark.DecodeCollector("http://local")
            with self.assertRaisesRegex(RuntimeError, "restarted"):
                collector.poll()


if __name__ == "__main__":
    unittest.main()
