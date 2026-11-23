"""Offline tiny-Qwen FP32 checks; set OMNISERVE_TEST_DEVICE=cuda for GPU tests."""

import importlib.util
import os
import unittest


HAS_MODEL_DEPENDENCIES = all(
    importlib.util.find_spec(package) is not None for package in ("torch", "transformers")
)


@unittest.skipUnless(HAS_MODEL_DEPENDENCIES, "requires torch and transformers; no models are downloaded")
class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from workers.decode_scheduler import Request, Scheduler

        cls.torch = torch
        cls.Request = Request
        cls.Scheduler = Scheduler
        cls.device = os.environ.get("OMNISERVE_TEST_DEVICE", "cpu")
        if cls.device not in {"cpu", "cuda"}:
            raise ValueError("OMNISERVE_TEST_DEVICE must be cpu or cuda")
        if cls.device == "cuda" and not torch.cuda.is_available():
            raise unittest.SkipTest("OMNISERVE_TEST_DEVICE=cuda requires an available CUDA device")
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(19)
        config = Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, bos_token_id=1, eos_token_id=None,
            pad_token_id=0, use_sliding_window=False,
        )
        config._attn_implementation = "eager"
        cls.model = Qwen2ForCausalLM(config).to(device=cls.device, dtype=torch.float32).eval()
        cls.model.generation_config.eos_token_id = None

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.model.generation_config.eos_token_id = None
        self.scheduler = self.Scheduler(self.model, self.device, max_active=2)

    def standalone(self, request):
        ids = self.torch.tensor([request.input_ids], device=self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=ids, attention_mask=self.torch.ones_like(ids),
                max_new_tokens=request.max_new_tokens, do_sample=False,
                pad_token_id=0, eos_token_id=None,
            )
        return output[0, len(request.input_ids):].tolist()

    def assert_cache_length(self, state):
        expected = len(state.request.input_ids) + len(state.generated) - 1
        self.assertEqual(state.cache.get_seq_length(), expected)
        for layer in state.cache.layers:
            self.assertEqual(layer.keys.shape[0], 1)
            self.assertEqual(layer.keys.shape[-2], expected)
            self.assertEqual(layer.values.shape[-2], expected)

    def test_shared_decode_of_unequal_prompts_matches_standalone(self):
        requests = [
            self.Request(0, [1, 4, 5, 6, 7], 5),
            self.Request(0, [1, 8], 5),
        ]
        expected = [self.standalone(request) for request in requests]
        states = [self.scheduler.start(request) for request in requests]
        for _ in range(3):
            self.scheduler.decode(states)
            for state, tokens in zip(states, expected, strict=True):
                self.assert_cache_length(state)
                self.assertEqual(state.generated, tokens[:len(state.generated)])
            for layers in zip(*(state.cache.layers for state in states), strict=True):
                self.assertNotEqual(
                    layers[0].keys.untyped_storage().data_ptr(),
                    layers[1].keys.untyped_storage().data_ptr(),
                )
                self.assertNotEqual(
                    layers[0].values.untyped_storage().data_ptr(),
                    layers[1].values.untyped_storage().data_ptr(),
                )
        self.scheduler.decode(states)
        self.assertEqual([state.generated for state in states], expected)
        self.assertTrue(all(self.scheduler.finished(state) for state in states))
        self.assertTrue(all(state.cache is None for state in states))

    def test_arrival_slot_reuse_and_single_shared_single_topology(self):
        from continuous_batching import run

        requests = [
            self.Request(0, [1, 4, 5], 1),
            self.Request(0, [1, 6], 4),
            self.Request(1, [1, 7, 8, 9], 6),
        ]
        tokens, trace = run(self.scheduler, requests)
        self.assertEqual(tokens, [self.standalone(request) for request in requests])
        self.assertIn(0, trace[0]["finished"])
        self.assertTrue(any(
            step["admitted"] == [2] and step["decode_request_ids"] == [1]
            for step in trace
        ))
        self.assertTrue(any(
            step["decode_request_ids"] == [1, 2]
            and len(set(step["decode_cache_lengths"])) == 2
            for step in trace
        ))
        self.assertTrue(any(step["decode_request_ids"] == [2] for step in trace))

    def test_one_token_budget_releases_prefill_cache(self):
        request = self.Request(0, [1, 4, 5], 1)
        state = self.scheduler.start(request)
        self.assertEqual(state.generated, self.standalone(request))
        self.assertTrue(self.scheduler.finished(state))
        self.assertIsNone(state.cache)

    def test_eos_releases_one_row_without_losing_other_cache(self):
        eos = 63
        self.model.generation_config.eos_token_id = eos
        scheduler = self.Scheduler(self.model, self.device, max_active=2)

        def force_tokens(module, args, kwargs, output):
            # Keep real attention and KV writes; control only the stopping tokens.
            output.logits.fill_(-1000)
            if kwargs.get("past_key_values") is None:
                output.logits[..., 5] = 0
            else:
                output.logits[0, -1, eos] = 0
                if output.logits.shape[0] == 2:
                    output.logits[1, -1, 6] = 0
            return output

        hook = self.model.register_forward_hook(force_tokens, with_kwargs=True)
        try:
            first = scheduler.start(self.Request(0, [1, 4, 5], 8))
            second = scheduler.start(self.Request(0, [1, 6], 8))
            scheduler.decode([first, second])
            self.assertEqual(first.generated, [5, eos])
            self.assertTrue(scheduler.finished(first))
            self.assertIsNone(first.cache)
            self.assertEqual(second.generated, [5, 6])
            self.assertFalse(scheduler.finished(second))
            self.assert_cache_length(second)
            scheduler.decode([second])
            self.assertEqual(second.generated, [5, 6, eos])
            self.assertTrue(scheduler.finished(second))
            self.assertIsNone(second.cache)
        finally:
            hook.remove()
            self.model.generation_config.eos_token_id = None

    def test_eos_during_prefill_releases_cache(self):
        request = self.Request(0, [1, 9, 10], 8)
        initial_token = self.scheduler.start(request).generated[0]
        self.scheduler.stops = {initial_token}
        state = self.scheduler.start(request)
        self.assertEqual(state.generated, [initial_token])
        self.assertTrue(self.scheduler.finished(state))
        self.assertIsNone(state.cache)


if __name__ == "__main__":
    unittest.main()
