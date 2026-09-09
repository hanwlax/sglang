"""FIAS V2 adapter for the existing expanded MHA prefill of MLA models."""


def fias_mha_prefill_v2(
    query,
    key,
    value,
    *,
    num_heads,
    input_layout,
    atten_mask,
    sparse_mode,
    scale,
    next_tokens,
    num_key_value_heads=0,
    query_rope=None,
    key_rope=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
):
    """Translate V1 keyword names only; do not transform Q/K/V or pad heads.

    Cached prefill retains BSND Q/K=192, V=128 with the true head count.
    Cold prefill retains the baseline TND split suffix and cumulative lengths.
    NZ cache has already been gathered/unpacked by the existing prefix path;
    this call consumes expanded ND tensors, not the paged latent cache.
    """
    import torch_npu

    return torch_npu.npu_fused_infer_attention_score_v2(
        query,
        key,
        value,
        num_query_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        input_layout=input_layout,
        atten_mask=atten_mask,
        sparse_mode=sparse_mode,
        softmax_scale=scale,
        next_tokens=next_tokens,
        query_rope=query_rope,
        key_rope=key_rope,
        actual_seq_qlen=(
            None if actual_seq_lengths is None else [int(n) for n in actual_seq_lengths]
        ),
        actual_seq_kvlen=(
            None
            if actual_seq_lengths_kv is None
            else [int(n) for n in actual_seq_lengths_kv]
        ),
    )
