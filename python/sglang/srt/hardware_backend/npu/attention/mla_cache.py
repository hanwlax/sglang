"""Read logical pages from the explicit PA-NZ storage of the NPU MLA cache."""

from functools import wraps

import torch
from sglang.srt.runtime_context import get_forward


def with_mla_nz_index_cache(forward):
    """Scope shared indices to one model call, including each capture warmup.

    NPUGraph records the index producers and re-executes them on replay.
    Never share this map between warmup and capture, even with the same loc.
    ContextVar access is deliberately excluded from torch.compile tracing.
    """

    @wraps(forward)
    def wrapped(*args, **kwargs):
        if torch.compiler.is_compiling():
            return forward(*args, **kwargs)
        with get_forward().scoped(npu_mla_nz_indices={}):
            return forward(*args, **kwargs)

    return wrapped


def get_mla_nz_indices(loc, page_size, head_dim, num_blocks):
    """Reuse same-stream indices while loc is immutable within a model call.

    Retain loc itself to prevent Python object-ID reuse. Versioned tensors
    invalidate on in-place writes; inference tensors have no version counter
    and must remain immutable within the forward, as required by attention.
    """
    from sglang.kernels.ops.kvcache.triton_mla_nz_indices import build_mla_nz_indices

    if torch.compiler.is_compiling():
        return build_mla_nz_indices(loc, page_size, head_dim, num_blocks)
    cache = get_forward().npu_mla_nz_indices
    if cache is None:
        return build_mla_nz_indices(loc, page_size, head_dim, num_blocks)
    version = None if loc.is_inference() else loc._version
    key = (
        id(loc),
        version,
        page_size,
        head_dim,
        num_blocks,
        torch.npu.current_stream(loc.device).npu_stream,
    )
    entry = cache.get(key)
    if entry is None:
        indices = build_mla_nz_indices(loc, page_size, head_dim, num_blocks)
        cache[key] = (loc, indices)
        return indices
    return entry[1]


def gather_mla_cache_pages(
    cache: torch.Tensor, block_ids: torch.Tensor, *, is_nz: bool
) -> torch.Tensor:
    """Return selected pages in logical [blocks, page_size, 1, head_dim] order.

    NZ buffers retain that public shape, but their physical contents are
    [blocks, head_dim // 16, page_size, 16]. Restore token-major order before
    projecting cached latent vectors or concatenating their RoPE features.
    """
    pages = torch.index_select(cache, 0, block_ids)
    if not is_nz:
        return pages
    page_size, head_dim = cache.shape[1], cache.shape[-1]
    return (
        pages.view(block_ids.numel(), head_dim // 16, page_size, 16)
        .permute(0, 2, 1, 3)
        .reshape(block_ids.numel(), page_size, 1, head_dim)
    )
