"""Exercise device selection and CUDA API calls without requiring a GPU wheel."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


WORKER = Path(__file__).resolve().parents[1] / "workers" / "llm_worker.py"


def load_worker(*, cuda=False, mps=False):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=Mock(return_value=cuda),
        synchronize=Mock(),
        reset_peak_memory_stats=Mock(),
        max_memory_allocated=Mock(return_value=7 * 2**20),
        max_memory_reserved=Mock(return_value=12 * 2**20),
        get_device_properties=Mock(return_value=SimpleNamespace(
            name="test GPU", total_memory=4 * 2**30,
        )),
    )
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=mps)))
    torch.mps = SimpleNamespace(synchronize=Mock())
    torch.version = SimpleNamespace(cuda="test CUDA")
    torch.float16, torch.float32 = object(), object()
    torch.inference_mode = lambda: lambda function: function
    transformers = ModuleType("transformers")
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.config = SimpleNamespace(_commit_hash="resolved-weights-commit")
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=Mock(return_value=model))
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock())
    specification = importlib.util.spec_from_file_location("isolated_llm_worker", WORKER)
    module = importlib.util.module_from_spec(specification)
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
        specification.loader.exec_module(module)
    return module, torch, transformers


class DeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        for cuda, mps, expected in (
            (True, True, "cuda"), (False, True, "mps"), (False, False, "cpu"),
        ):
            with self.subTest(cuda=cuda, mps=mps):
                worker, _, _ = load_worker(cuda=cuda, mps=mps)
                self.assertEqual(worker.resolve_device("auto"), expected)

    def test_explicit_unavailable_gpu_fails_instead_of_falling_back(self):
        worker, _, _ = load_worker()
        for device in ("cuda", "mps", "invalid"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                worker.resolve_device(device)
        self.assertEqual(worker.resolve_device("cpu"), "cpu")

    def test_cuda_uses_half_precision_and_pins_tokenizer_to_resolved_weights(self):
        worker, torch, transformers = load_worker(cuda=True)
        arguments = SimpleNamespace(
            device="auto", model=worker.MODEL_ID, revision=None, local_files_only=True,
        )
        model, _, revision = worker.load_model(arguments)
        self.assertEqual(arguments.device, "cuda")
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            worker.MODEL_ID, revision=worker.MODEL_REVISION,
            local_files_only=True, dtype=torch.float16,
        )
        model.to.assert_called_once_with("cuda")
        model.eval.assert_called_once_with()
        self.assertEqual(revision, "resolved-weights-commit")
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            worker.MODEL_ID, revision=revision, local_files_only=True,
        )

    def test_cpu_keeps_full_precision(self):
        worker, torch, transformers = load_worker()
        arguments = SimpleNamespace(
            device="cpu", model="custom/model", revision="requested", local_files_only=False,
        )
        worker.load_model(arguments)
        self.assertIs(
            transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["dtype"],
            torch.float32,
        )

    def test_cuda_timing_waits_for_device_and_reads_peak_counters(self):
        worker, torch, _ = load_worker(cuda=True)
        worker.synchronize("cuda")
        torch.cuda.synchronize.assert_called_once_with("cuda")
        torch.mps.synchronize.assert_not_called()
        worker.reset_peak_memory("cuda")
        torch.cuda.reset_peak_memory_stats.assert_called_once_with("cuda")
        self.assertEqual(worker.peak_memory("cuda"), {
            "cuda_peak_allocated_mb": 7.0, "cuda_peak_reserved_mb": 12.0,
        })
        self.assertEqual(worker.device_metadata("cuda"), {
            "cuda_version": "test CUDA", "cuda_device_name": "test GPU",
            "cuda_total_memory_mb": 4096.0,
        })

    def test_cpu_and_mps_do_not_query_cuda_allocator(self):
        worker, torch, _ = load_worker(mps=True)
        worker.synchronize("cpu")
        worker.synchronize("mps")
        torch.mps.synchronize.assert_called_once_with()
        torch.cuda.synchronize.assert_not_called()
        for device in ("cpu", "mps"):
            self.assertEqual(worker.peak_memory(device), {})
            self.assertEqual(worker.device_metadata(device), {})
            worker.reset_peak_memory(device)
        torch.cuda.reset_peak_memory_stats.assert_not_called()
        torch.cuda.max_memory_allocated.assert_not_called()
        torch.cuda.max_memory_reserved.assert_not_called()
        torch.cuda.get_device_properties.assert_not_called()

    def test_command_line_accepts_cuda(self):
        worker, _, _ = load_worker()
        with patch.object(sys, "argv", ["llm_worker.py", "--device", "cuda"]):
            self.assertEqual(worker.parse_args().device, "cuda")


if __name__ == "__main__":
    unittest.main()
