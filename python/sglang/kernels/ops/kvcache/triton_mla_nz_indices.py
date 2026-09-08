"""Fuse explicit PA-NZ cache indexing; retain native cache scatter writes."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _mla_nz_indices_kernel(
    loc,
    out,
    N,
    PAGE_SIZE: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    TILES: tl.constexpr,
    LOC_STRIDE: tl.constexpr,
    INT32_INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(loc + (idx // TILES) * LOC_STRIDE, idx < N * TILES, other=0)
    if INT32_INDEX:
        slot = slot.to(tl.int32)
    else:
        slot = slot.to(tl.int64)
    if PAGE_SIZE == (1 << PAGE_SHIFT):
        # Ascend's int32 division lowering can round large slots through fp32.
        # Slots are nonnegative: shifts/masks preserve exact page coordinates.
        page = slot >> PAGE_SHIFT
        in_page = slot & (PAGE_SIZE - 1)
    else:
        page = slot.to(tl.int64) // PAGE_SIZE
        in_page = slot.to(tl.int64) % PAGE_SIZE
    offset = page * (TILES * PAGE_SIZE) + (idx % TILES) * PAGE_SIZE + in_page
    tl.store(out + idx, offset.to(tl.int64), idx < N * TILES)


def build_mla_nz_indices(loc, page_size, head_dim, num_blocks):
    """Return 16-element-row indices for [page, dim//16, page_size, 16].

    Device locations remain runtime inputs to graph replay. The token count is
    a non-specialized scalar, so different prefill tails share one binary for
    each layout/index-width combination. Valid slots fit the supplied cache;
    retain int64 arithmetic when its row indices can exceed signed int32.
    """
    if head_dim <= 0 or head_dim % 16:
        raise ValueError("PA-NZ MLA cache head dimension must be divisible by 16")
    if page_size <= 0:
        raise ValueError("PA-NZ MLA cache page size must be positive")
    loc = loc.reshape(-1)
    tiles = head_dim // 16
    indices = torch.empty((loc.numel() * tiles,), device=loc.device, dtype=torch.int64)
    if loc.numel():
        _mla_nz_indices_kernel[(triton.cdiv(indices.numel(), 256),)](
            loc,
            indices,
            loc.numel(),
            page_size,
            page_size.bit_length() - 1,
            tiles,
            loc.stride(0),
            num_blocks * page_size * tiles <= 2**31,
            256,
        )
    return indices
