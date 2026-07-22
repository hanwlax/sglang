from typing import Optional


import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu

if not is_cpu():
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )
    from sglang.srt.layers.attention.fla.fused_recurrent_linear_replayssm import (
        fused_recurrent_linear_replayssm_decode,
    )
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.srt.layers.attention.fla.kda import chunk_kda

if is_npu():
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu

import os
_KDA_USE_TORCH_NATIVE_EXTEND = os.getenv("SGLANG_KDA_TORCH_NATIVE_EXTEND", "0") == "1"

def kda_extend_torch_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float = None,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    A_log: torch.Tensor = None,
    dt_bias: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor = None,
) -> torch.Tensor:
    """Torch-native KDA prefill (sequential recurrence).

    Replaces ``chunk_kda`` / ``fused_recurrent_kda`` with a pure PyTorch
    per-token loop.  Correctness reference on Ascend where the Triton
    JIT may produce zero-valued or silently incorrect results.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    B = q.shape[0]
    T = q.shape[1]
    H = q.shape[2]
    K_dim = q.shape[3]
    HV = v.shape[2]
    V_dim = v.shape[3]

    out_dtype = v.dtype
    q = q.float()
    k = k.float()
    v = v.float()
    beta = beta.float()
    A_log = A_log.float().view(1, 1, -1, 1)
    dt_bias = dt_bias.float().view(1, 1, -1, K_dim)

    if use_qk_l2norm_in_kernel:
        q = q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-6)
    q = q * scale

    # Activate gate: g_act = -exp(A_log) * softplus(g + dt_bias)
    gate_x = g.float() + dt_bias
    gate_act = -torch.exp(A_log) * F.softplus(gate_x, beta=1.0, threshold=20.0)
    gate_exp = torch.exp(gate_act)  # [1, T, H, K]

    ssm_pool = initial_state
    out = torch.empty(B, T, HV, V_dim, device=q.device, dtype=q.dtype)

    if cu_seqlens is not None and cu_seqlens.shape[0] > 2:
        seq_starts = cu_seqlens[:-1]
    else:
        seq_starts = torch.tensor([0], dtype=torch.long, device=cu_seqlens.device if cu_seqlens is not None else "cpu")

    for seq_i, start in enumerate(seq_starts):
        if cu_seqlens is not None:
            end = cu_seqlens[seq_i + 1]
        else:
            end = B * T

        idx = initial_state_indices[seq_i].item()
        if idx >= 0:
            state = ssm_pool[idx].float()  # [HV, V, K]
        else:
            state = torch.zeros(HV, V_dim, K_dim, device=q.device)

        gqa_ratio = HV // H
        for t in range(start, end):
            qt = q[0, t]  # [H, K]
            kt = k[0, t]  # [H, K]
            vt = v[0, t]  # [HV, V]
            ge = gate_exp[0, t]  # [H, K]
            bt = beta[0, t]  # [HV]

            if gqa_ratio > 1:
                kt = kt.repeat_interleave(gqa_ratio, dim=0)
                qt = qt.repeat_interleave(gqa_ratio, dim=0)
                ge = ge.repeat_interleave(gqa_ratio, dim=0)

            state = state * ge.unsqueeze(1)
            v_upd = vt - (state @ kt.unsqueeze(-1)).squeeze(-1)
            v_upd = v_upd * bt.unsqueeze(-1)
            state = state + v_upd.unsqueeze(-1) * kt.unsqueeze(1)
            ot = (state @ qt.unsqueeze(-1)).squeeze(-1)
            out[0, t] = ot.to(out.dtype)

        if idx >= 0:
            ssm_pool[idx] = state.to(ssm_pool.dtype)

    return out.to(out_dtype)


class TritonKDAKernel(LinearAttnKernelBase):
    """Triton-based kernel for KDA (Kimi Delta Attention) linear attention."""

    supports_packed_decode: bool = not is_cpu() and not is_npu()

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: feed the conv-1d output ``mixed_qkv``
        straight into a single fused Triton kernel that does Q/K/V extraction,
        gate/beta computation, l2-norm, and the recurrent state update.

        Returns output tensor of shape [1, B, HV, V] to match the existing
        decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        # KDA ReplaySSM buffered decode: drop-in for the packed decode, same
        # args plus the three per-layer ring caches + the per-row write cursor
        # (and optional radix-track force-flush). Uses the gate-generic kernel
        # with is_kda=True (per-K gate); g_cache is [num_slots, HV, L, K].
        # When any ring tensor / cursor is None (flag off) we fall through to
        # the byte-identical legacy path below.
        replayssm_d = kwargs.get("replayssm_d")
        replayssm_k = kwargs.get("replayssm_k")
        replayssm_g = kwargs.get("replayssm_g")
        replayssm_write_pos = kwargs.get("replayssm_write_pos")
        replayssm_force_flush = kwargs.get("replayssm_force_flush")
        if (
            replayssm_d is not None
            and replayssm_k is not None
            and replayssm_g is not None
            and replayssm_write_pos is not None
        ):
            K = ssm_states.shape[-1]  # ssm_states: [num_slots, HV, V, K]
            fused_recurrent_linear_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a.reshape(B, num_v_heads, K).contiguous(),
                b=b.reshape(B, num_v_heads).contiguous(),
                A_log=A_log.reshape(-1),
                dt_bias=dt_bias.reshape(num_v_heads, K).contiguous(),
                scale=scale,
                initial_state=ssm_states,
                d_cache=replayssm_d,
                k_cache=replayssm_k,
                g_cache=replayssm_g,
                out=out,
                ssm_state_indices=cache_indices,
                write_pos=replayssm_write_pos,
                force_flush=replayssm_force_flush,
                use_qk_l2norm_in_kernel=True,
                is_kda=True,
            )
            return out.transpose(0, 1)

        # a may come in as [B, HV, K] (or [B, 1, HV*K]); b may come in as
        # [B, 1, HV]. Flatten both to the 2D shapes the kernel expects.
        if a.dim() != 2:
            a = a.reshape(B, -1)
        if b.dim() != 2:
            b = b.reshape(B, -1)
        fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log.reshape(-1),
            dt_bias=dt_bias.reshape(-1),
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )
        # [B, 1, HV, V] -> [1, B, HV, V] view to match existing decode layout.
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        if _KDA_USE_TORCH_NATIVE_EXTEND and A_log is not None:
            return kda_extend_torch_native(
                q=q, k=k, v=v, g=g, beta=beta,
                scale=None,
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
                A_log=A_log,
                dt_bias=dt_bias,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc,
            )
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )
