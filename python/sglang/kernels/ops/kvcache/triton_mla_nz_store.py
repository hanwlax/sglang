"""Write latent and RoPE values directly into explicit PA-NZ cache storage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_nz_store_kernel(
    loc,
    key,
    rope,
    key_cache,
    rope_cache,
    LOC_STRIDE: tl.constexpr,
    K_ROW_STRIDE: tl.constexpr,
    K_COL_STRIDE: tl.constexpr,
    R_ROW_STRIDE: tl.constexpr,
    R_COL_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    K_DIM: tl.constexpr,
    R_DIM: tl.constexpr,
    K_BLOCK: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    # Element addresses can exceed int32 even when 16-element row indices fit.
    slot = tl.load(loc + token * LOC_STRIDE).to(tl.int64)
    if PAGE_SIZE == (1 << PAGE_SHIFT):
        page, in_page = slot >> PAGE_SHIFT, slot & (PAGE_SIZE - 1)
    else:
        page, in_page = slot // PAGE_SIZE, slot % PAGE_SIZE

    k = tl.arange(0, K_BLOCK)
    k_value = tl.load(key + token * K_ROW_STRIDE + k * K_COL_STRIDE, k < K_DIM, other=0)
    k_dst = tl.make_block_ptr(
        base=key_cache + page * (K_DIM * PAGE_SIZE) + in_page * 16,
        shape=(K_DIM // 16, 16),
        strides=(PAGE_SIZE * 16, 1),
        offsets=(0, 0),
        block_shape=(K_BLOCK // 16, 16),
        order=(1, 0),
    )
    tl.store(k_dst, k_value.reshape(K_BLOCK // 16, 16), boundary_check=(0,))

    r = tl.arange(0, R_BLOCK)
    r_value = tl.load(
        rope + token * R_ROW_STRIDE + r * R_COL_STRIDE, r < R_DIM, other=0
    )
    r_dst = tl.make_block_ptr(
        base=rope_cache + page * (R_DIM * PAGE_SIZE) + in_page * 16,
        shape=(R_DIM // 16, 16),
        strides=(PAGE_SIZE * 16, 1),
        offsets=(0, 0),
        block_shape=(R_BLOCK // 16, 16),
        order=(1, 0),
    )
    tl.store(r_dst, r_value.reshape(R_BLOCK // 16, 16), boundary_check=(0,))


def store_mla_nz_cache(loc, key, rope, key_cache, rope_cache, page_size):
    """Copy token rows without materializing indices or packing source values.

    Destinations are contiguous [pages, page_size, 1, dim] allocations whose
    physical order is [page, dim//16, slot, 16]. Live locations must be unique,
    in range, and disjoint from source storage, as for native scatter writes.
    The caller performs dtype conversion before entering this function.
    """
    loc = loc.reshape(-1)
    count = loc.numel()
    if not count:
        return
    k_dim, r_dim = key_cache.shape[-1], rope_cache.shape[-1]
    if page_size <= 0 or any(d <= 0 or d % 16 for d in (k_dim, r_dim)):
        raise ValueError("PA-NZ requires a positive page size and dims divisible by 16")
    if not key_cache.is_contiguous() or not rope_cache.is_contiguous():
        raise ValueError("PA-NZ destination allocations must be contiguous")
    if any(x.dtype != key.dtype for x in (rope, key_cache, rope_cache)):
        raise ValueError("PA-NZ source and destination dtypes must match")
    if key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Direct PA-NZ store supports fp16 and bf16 storage")
    key, rope = key.reshape(count, k_dim), rope.reshape(count, r_dim)
    _mla_nz_store_kernel[(count,)](
        loc,
        key,
        rope,
        key_cache,
        rope_cache,
        loc.stride(0),
        *key.stride(),
        *rope.stride(),
        page_size,
        page_size.bit_length() - 1,
        k_dim,
        r_dim,
        triton.next_power_of_2(k_dim),
        triton.next_power_of_2(r_dim),
    )
