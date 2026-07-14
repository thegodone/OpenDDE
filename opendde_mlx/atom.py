"""Atom-level local windowed attention primitives for OpenDDE in MLX.

The trunk uses sliding-window local attention: each block of n_queries=32 atoms
attends to a centered window of n_keys=128 atoms. This mirrors
opendde.model.modules.primitives.rearrange_qk_to_dense_trunk (torch .unfold)
using pad + gather (MLX has no unfold).
"""

from __future__ import annotations

import math

import mlx.core as mx


def rearrange_qk_to_dense_trunk(
    q: mx.array,
    k: mx.array,
    n_queries: int = 32,
    n_keys: int = 128,
):
    """q,k: [..., N_atom, C] (atom axis = -2).

    Returns:
      q_trunked: [..., n_trunks, n_queries, C]
      k_trunked: [..., n_trunks, n_keys,  C]
      mask_trunked (bool): [n_trunks, n_queries, n_keys]
      info: dict(q_pad, pad_left, pad_right, n_trunks)
    """
    assert n_keys >= n_queries and n_queries % 2 == 0 and n_keys % 2 == 0
    n = q.shape[-2]
    c = q.shape[-1]
    lead = q.shape[:-2]

    n_trunks = int(math.ceil(n / n_queries))
    q_pad = n_trunks * n_queries - n
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 1 / 2) * n_queries + n_keys / 2 - n + 1 / 2)

    # --- queries: pad on the right, reshape into [n_trunks, n_queries] ---
    if q_pad:
        qpad = mx.zeros((*lead, q_pad, c), dtype=q.dtype)
        qp = mx.concatenate([q, qpad], axis=-2)
    else:
        qp = q
    q_trunked = qp.reshape(*lead, n_trunks, n_queries, c)

    # --- keys: pad (pad_left, pad_right), gather sliding windows ---
    kp = mx.concatenate([
        mx.zeros((*lead, pad_left, c), dtype=k.dtype),
        k,
        mx.zeros((*lead, pad_right, c), dtype=k.dtype),
    ], axis=-2)
    # window t covers padded positions [t*n_queries : t*n_queries + n_keys]
    idx = (mx.arange(n_trunks)[:, None] * n_queries
           + mx.arange(n_keys)[None, :])            # [n_trunks, n_keys]
    k_trunked = mx.take(kp, idx, axis=-2)           # [..., n_trunks, n_keys, C]

    # --- validity mask (bool): real query AND key maps to a real (non-pad) atom ---
    q_global = mx.arange(n_trunks)[:, None] * n_queries + mx.arange(n_queries)[None, :]
    q_real = q_global < n                                   # [n_trunks, n_queries]
    key_real = (idx >= pad_left) & (idx < pad_left + n)      # [n_trunks, n_keys]
    mask_trunked = q_real[:, :, None] & key_real[:, None, :]  # [n_trunks, n_q, n_k]

    info = dict(q_pad=q_pad, pad_left=pad_left, pad_right=pad_right, n_trunks=n_trunks)
    return q_trunked, k_trunked, mask_trunked, info


def local_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    n_queries: int = 32,
    n_keys: int = 128,
    trunked_attn_bias: mx.array | None = None,
    inf: float = 1e10,
) -> mx.array:
    """Windowed local attention. q,k,v: [..., N, d] (q already scaled by caller).
    trunked_attn_bias: [..., n_trunks, n_queries, n_keys] additive pair bias.
    Matches opendde primitives._local_attention (torch)."""
    n = q.shape[-2]
    qt, _, _, info = rearrange_qk_to_dense_trunk(q, q, n_queries, n_keys)
    _, kt, mask, _ = rearrange_qk_to_dense_trunk(k, k, n_queries, n_keys)
    _, vt, _, _ = rearrange_qk_to_dense_trunk(v, v, n_queries, n_keys)
    # window mask -> additive bias (0 valid, -inf invalid); broadcasts over lead/head dims
    bias = mx.where(mask, mx.array(0.0, q.dtype), mx.array(-inf, q.dtype))
    if trunked_attn_bias is not None:
        bias = bias + trunked_attn_bias
    scores = mx.einsum("...qd,...kd->...qk", qt, kt) + bias      # [..., nt, nq, nk]
    attn = mx.softmax(scores, axis=-1)
    out = mx.einsum("...qk,...kd->...qd", attn, vt)              # [..., nt, nq, d]
    out = out.reshape(*out.shape[:-3], -1, out.shape[-1])        # [..., nt*nq, d]
    return out[..., :n, :]
