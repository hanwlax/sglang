"""Variable-length K3 prefill directly over the paged MLA latent cache."""

from collections.abc import Sequence

import torch


def fias_v2_mla_prefill(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    cache: torch.Tensor,
    rope_cache: torch.Tensor,
    *,
    query_lens: Sequence[int],
    kv_lens: Sequence[int],
    block_table: torch.Tensor,
    page_size: int,
    scale: float,
    mask: torch.Tensor,
    is_nz: bool,
) -> torch.Tensor:
    """Return [tokens, local_heads, latent_dim], with DP padding zeroed.

    Use one BNSD call per non-empty request so ragged query lengths require
    neither query padding to the longest prefill nor cumulative KV lengths.
    sparse_mode=3 right-aligns the causal mask with the cached prefix. Unlike
    target verify, query lengths come from this chunk, not the draft block.
    Cache writes must be completed on the current stream before calling this.
    """
    import torch_npu

    if (
        query.ndim != 3
        or query_rope.ndim != 3
        or query_rope.shape[:2] != query.shape[:2]
    ):
        raise ValueError(
            "MLA prefill expects matching token/head axes for Q and Q_rope"
        )
    tokens, heads, latent_dim = query.shape
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FIAS V2 prefill requires FP16/BF16 query and KV cache")
    if any(t.dtype != query.dtype for t in (query_rope, cache, rope_cache)):
        raise ValueError("FIAS V2 prefill does not support quantized KV cache")
    if latent_dim != 512 or query_rope.shape[-1] != 64 or not 1 <= heads <= 128:
        raise ValueError("K3 paged MLA prefill requires D=512, rope D=64, 1..128 heads")
    if not 0 < page_size <= 1024 or page_size % 16:
        raise ValueError("MLA page_size must be 16-aligned and at most 1024")
    for buf, dim in ((cache, latent_dim), (rope_cache, query_rope.shape[-1])):
        if buf.ndim != 4 or tuple(buf.shape[1:]) != (page_size, 1, dim):
            raise ValueError(
                "MLA cache must have logical shape [pages, page_size, 1, D]"
            )
    if cache.shape[0] != rope_cache.shape[0]:
        raise ValueError("Latent and RoPE cache must have the same page count")
    if (
        block_table.ndim != 2
        or len(query_lens) != len(kv_lens)
        or block_table.shape[0] != len(query_lens)
    ):
        raise ValueError("Query lengths, KV lengths and block-table rows must match")
    if sum(query_lens) > tokens:
        raise ValueError("Prefill query lengths exceed the query token count")
    for q_len, kv_len in zip(query_lens, kv_lens):
        if q_len < 0 or kv_len < q_len or kv_len > block_table.shape[1] * page_size:
            raise ValueError("Invalid prefill query/KV length or truncated block table")

    # Public cache shapes stay token-major even when the physical contents
    # are NZ. Do NOT permute or gather here: view the persistent storage.
    if is_nz:
        key = cache.view(-1, 1, latent_dim // 16, page_size, 16)
        key_rope = rope_cache.view(-1, 1, 4, page_size, 16)
    else:
        key = cache.view(-1, 1, page_size, latent_dim)
        key_rope = rope_cache.view(-1, 1, page_size, 64)

    padded_heads = 1 << (heads - 1).bit_length()
    output = torch.zeros_like(query)
    offset = 0
    for row, (q_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        if q_len == 0:
            continue
        q = query[offset : offset + q_len].transpose(0, 1).unsqueeze(0)
        q_rope = query_rope[offset : offset + q_len].transpose(0, 1).unsqueeze(0)
        if padded_heads != heads:
            # Padding heads must be initialized; they do not share softmax
            # with the real heads and are discarded after the call.
            q = torch.cat(
                (q, q.new_zeros(1, padded_heads - heads, q_len, latent_dim)), dim=1
            )
            q_rope = torch.cat(
                (q_rope, q_rope.new_zeros(1, padded_heads - heads, q_len, 64)), dim=1
            )
        result, _ = torch_npu.npu_fused_infer_attention_score_v2(
            q.contiguous(),
            key,
            key,
            query_rope=q_rope.contiguous(),
            key_rope=key_rope,
            num_query_heads=padded_heads,
            num_key_value_heads=1,
            input_layout="BNSD",
            softmax_scale=scale,
            block_table=block_table[row : row + 1].contiguous(),
            block_size=page_size,
            sparse_mode=3,
            atten_mask=mask,
            actual_seq_qlen=[q_len],
            actual_seq_kvlen=[kv_len],
            pre_tokens=2147483647,
            next_tokens=0,
        )
        output[offset : offset + q_len] = result[0, :heads].transpose(0, 1)
        offset += q_len
    return output
