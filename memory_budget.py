"""Pure memory estimates for a serial CUDA generation worker."""


def estimate_kv_bytes(*, layers, kv_heads, head_dim, tokens, element_bytes):
    """K and V use [batch=1, kv_heads, tokens, head_dim] per layer."""
    if min(layers, kv_heads, head_dim, element_bytes) < 1 or tokens < 0:
        raise ValueError("Invalid KV cache dimensions.")
    return 2 * layers * kv_heads * head_dim * tokens * element_bytes


def cuda_admission_budget(*, kv_bytes, input_tokens, attention_heads,
                          driver_free_bytes, total_bytes, reserved_bytes, allocated_bytes):
    # Cached allocator blocks belong to this process and can be reused. Counting
    # only driver-free memory can reject requests after an otherwise idle run.
    reusable = max(0, reserved_bytes - allocated_bytes)
    available = driver_free_bytes + reusable
    workspace = 2 * 4 * attention_heads * input_tokens ** 2
    headroom = max(512 * 1024 ** 2, total_bytes // 10)
    required = 2 * kv_bytes + workspace + headroom
    return {
        "cuda_driver_free_bytes": driver_free_bytes,
        "cuda_reusable_reserved_bytes": reusable,
        "cuda_free_bytes": available,
        "cuda_required_bytes": required,
        "cuda_admitted": required <= available,
    }
