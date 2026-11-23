import unittest

from memory_budget import cuda_admission_budget, estimate_kv_bytes


class MemoryBudgetTests(unittest.TestCase):
    def test_gqa_uses_kv_heads_and_counts_both_keys_and_values(self):
        # Qwen2.5-0.5B: 24 layers, 2 KV heads, head_dim=64, FP16.
        self.assertEqual(estimate_kv_bytes(layers=24, kv_heads=2, head_dim=64,
                                          tokens=100, element_bytes=2), 1_228_800)

    def test_allocator_cache_is_available_to_following_request(self):
        budget = cuda_admission_budget(kv_bytes=1024, input_tokens=16,
                                       attention_heads=14, driver_free_bytes=128 * 2**20,
                                       total_bytes=4 * 2**30, reserved_bytes=1024 * 2**20,
                                       allocated_bytes=256 * 2**20)
        self.assertTrue(budget["cuda_admitted"])
        self.assertEqual(budget["cuda_free_bytes"], 896 * 2**20)

    def test_insufficient_memory_is_rejected_without_allocating(self):
        budget = cuda_admission_budget(kv_bytes=1024, input_tokens=16,
                                       attention_heads=14, driver_free_bytes=0,
                                       total_bytes=4 * 2**30, reserved_bytes=0,
                                       allocated_bytes=0)
        self.assertFalse(budget["cuda_admitted"])


if __name__ == "__main__":
    unittest.main()
