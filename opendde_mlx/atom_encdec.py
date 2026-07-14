"""MLX port of OpenDDE atom encoder/decoder (AF3 Alg. 5/6/7).

Ports:
  - AtomAttentionEncoder  (opendde/model/modules/transformer.py:1213)
  - AtomAttentionDecoder  (:2171)
  - AtomTransformer       (:1067) -> DiffusionTransformer (cross_attention_mode)
      -> DiffusionTransformerBlock -> AttentionPairBias (local, has_s) +
         ConditionedTransitionBlock.

Reuses opendde_mlx/atom.py windowed local attention and opendde_mlx/modules.py
elementary primitives. Weight-key names mirror the torch submodule tree so the
checkpoint subtree loads 1:1 (only remapping is `_sub` prefix stripping).

The encoder here implements the `has_coords=False` input-embedder path (r_l/s/z
are None): prepare_cache builds c_l and p_lm from the reference features, the
atom transformer runs windowed cross attention, and per-atom features are mean
aggregated to per-token `a`.
"""

from __future__ import annotations

import math

import mlx.core as mx

from opendde_mlx.atom import local_attention, rearrange_qk_to_dense_trunk
from opendde_mlx.modules import _sub, layer_norm, linear, silu


# ---------------------------------------------------------------------------
# AdaptiveLayerNorm (AF3 Alg. 26) — primitives.py:96
# ---------------------------------------------------------------------------
def adaptive_layer_norm(a: mx.array, s: mx.array, p: dict, ln_eps: float = 1e-5) -> mx.array:
    """a_n = LN(a) [no scale/offset]; s_n = LN(s) [scale only];
    out = sigmoid(linear_s(s_n)) * a_n + linear_nobias_s(s_n)."""
    a_n = layer_norm(a, None, None, ln_eps)
    s_n = layer_norm(s, p["layernorm_s.weight"], None, ln_eps)
    gate = mx.sigmoid(linear(s_n, p["linear_s.weight"], p["linear_s.bias"]))
    return gate * a_n + linear(s_n, p["linear_nobias_s.weight"])


# ---------------------------------------------------------------------------
# AttentionPairBias — local, cross-attention, has_s  (transformer.py:39)
# ---------------------------------------------------------------------------
def attention_pair_bias_local(
    a: mx.array,
    s: mx.array,
    z: mx.array,
    p: dict,
    n_heads: int,
    n_queries: int,
    n_keys: int,
    inf: float = 1e10,
    ln_eps: float = 1e-5,
) -> mx.array:
    """Local (windowed) AttentionPairBias used by AtomTransformer.

    a,s: [..., N_atom, c_a]; z: [..., n_blocks, n_queries, n_keys, c_z].
    Returns the block output (already gated by sigmoid(linear_a_last(s))).
    """
    # Line2 (AdaLN) — query and key/value normed separately (cross_attention_mode)
    a_n = adaptive_layer_norm(a, s, _sub(p, "layernorm_a"), ln_eps)
    kv = adaptive_layer_norm(a_n, s, _sub(p, "layernorm_kv"), ln_eps)

    # Pair bias: linear(LN(z)) -> [..., n_blocks, n_q, n_keys, n_heads]
    zn = layer_norm(z, p["layernorm_z.weight"], None, ln_eps)
    bias = linear(zn, p["linear_nobias_z.weight"])
    # permute_final_dims [3,0,1,2]: -> [..., n_heads, n_blocks, n_q, n_keys]
    bias = mx.moveaxis(bias, -1, -4)

    # QKV projection (attention submodule)
    ap = _sub(p, "attention")
    q = linear(a_n, ap["linear_q.weight"], ap["linear_q.bias"])
    k = linear(kv, ap["linear_k.weight"])
    v = linear(kv, ap["linear_v.weight"])

    def to_heads(t):
        t = t.reshape(*t.shape[:-1], n_heads, -1)  # [..., N, H, ch]
        return mx.moveaxis(t, -2, -3)              # [..., H, N, ch]

    q = to_heads(q)
    k = to_heads(k)
    v = to_heads(v)
    c_hidden = q.shape[-1]
    q = q / math.sqrt(c_hidden)

    o = local_attention(q, k, v, n_queries, n_keys, trunked_attn_bias=bias, inf=inf)
    o = mx.moveaxis(o, -3, -2)  # [..., N, H, ch]

    # gating on q_x (a_n)
    g = mx.sigmoid(linear(a_n, ap["linear_g.weight"]))
    g = g.reshape(*g.shape[:-1], n_heads, -1)
    o = o * g
    o = o.reshape(*o.shape[:-2], -1)  # flatten heads
    o = linear(o, ap["linear_o.weight"])

    # adaLN-Zero output gate
    return mx.sigmoid(linear(s, p["linear_a_last.weight"], p["linear_a_last.bias"])) * o


# ---------------------------------------------------------------------------
# ConditionedTransitionBlock (AF3 Alg. 25) — transformer.py:1166
# ---------------------------------------------------------------------------
def conditioned_transition_block(a: mx.array, s: mx.array, p: dict, ln_eps: float = 1e-5) -> mx.array:
    a = adaptive_layer_norm(a, s, _sub(p, "adaln"), ln_eps)
    b = silu(linear(a, p["linear_nobias_a1.weight"])) * linear(a, p["linear_nobias_a2.weight"])
    return mx.sigmoid(linear(s, p["linear_s.weight"], p["linear_s.bias"])) * linear(
        b, p["linear_nobias_b.weight"]
    )


# ---------------------------------------------------------------------------
# DiffusionTransformerBlock — transformer.py:754
# ---------------------------------------------------------------------------
def diffusion_transformer_block(
    a: mx.array,
    s: mx.array,
    z: mx.array,
    p: dict,
    n_heads: int,
    n_queries: int,
    n_keys: int,
    ln_eps: float = 1e-5,
) -> mx.array:
    attn_out = attention_pair_bias_local(
        a, s, z, _sub(p, "attention_pair_bias"), n_heads, n_queries, n_keys, ln_eps=ln_eps
    )
    attn_out = attn_out + a
    ff_out = conditioned_transition_block(attn_out, s, _sub(p, "conditioned_transition_block"), ln_eps)
    return ff_out + attn_out


def atom_transformer(
    q: mx.array,
    c: mx.array,
    p_lm: mx.array,
    p: dict,
    n_blocks: int = 3,
    n_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    ln_eps: float = 1e-5,
) -> mx.array:
    """AtomTransformer (AF3 Alg. 7): windowed DiffusionTransformer with a=q, s=c, z=p_lm.
    `p` is the atom_transformer.diffusion_transformer subtree."""
    a = q
    for i in range(n_blocks):
        a = diffusion_transformer_block(
            a, c, p_lm, _sub(p, f"blocks.{i}"), n_heads, n_queries, n_keys, ln_eps
        )
    return a


# ---------------------------------------------------------------------------
# token <-> atom helpers (model/utils.py)
# ---------------------------------------------------------------------------
def broadcast_token_to_atom(x_token: mx.array, atom_to_token_idx: mx.array) -> mx.array:
    """x_token: [..., N_token, d]; atom_to_token_idx: [N_atom] -> [..., N_atom, d]."""
    return mx.take(x_token, atom_to_token_idx, axis=-2)


def aggregate_atom_to_token_mean(
    x_atom: mx.array, atom_to_token_idx: mx.array, n_token: int | None = None
) -> mx.array:
    """Mean-scatter atoms into tokens along axis -2. atom_to_token_idx: [N_atom]."""
    idx = atom_to_token_idx.astype(mx.int32)
    if n_token is None:
        n_token = int(idx.max().item()) + 1
    lead = x_atom.shape[:-2]
    d = x_atom.shape[-1]
    onehot = (idx[:, None] == mx.arange(n_token)[None, :]).astype(x_atom.dtype)  # [N_atom, N_token]
    # sums: [..., N_token, d]
    sums = mx.einsum("...ad,at->...td", x_atom, onehot)
    counts = onehot.sum(axis=0)  # [N_token]
    counts = mx.maximum(counts, mx.array(1.0, x_atom.dtype))
    counts = counts.reshape(*([1] * len(lead)), n_token, 1)
    return sums / counts


# ---------------------------------------------------------------------------
# AtomAttentionEncoder — transformer.py:1213 (has_coords=False path)
# ---------------------------------------------------------------------------
def atom_attention_encoder_prepare_cache(
    ref_pos: mx.array,
    ref_charge: mx.array,
    ref_mask: mx.array,
    ref_element: mx.array,
    ref_atom_name_chars: mx.array,
    d_lm: mx.array,
    v_lm: mx.array,
    mask_trunked: mx.array,
    p: dict,
) -> tuple[mx.array, mx.array]:
    """Returns (p_lm, c_l). Mirrors AtomAttentionEncoder.prepare_cache (r_l=None)."""
    batch_shape = ref_pos.shape[:-2]
    N_atom = ref_pos.shape[-2]

    c_l = linear(ref_pos, p["linear_no_bias_ref_pos.weight"]) + linear(
        mx.arcsinh(ref_charge).reshape(*batch_shape, N_atom, 1),
        p["linear_no_bias_ref_charge.weight"],
    )
    feat = mx.concatenate(
        [
            ref_mask.reshape(*batch_shape, N_atom, 1),
            ref_element.reshape(*batch_shape, N_atom, 128),
            ref_atom_name_chars.reshape(*batch_shape, N_atom, 4 * 64),
        ],
        axis=-1,
    )
    c_l = c_l + linear(feat, p["linear_no_bias_f.weight"])
    c_l = c_l * ref_mask.reshape(*batch_shape, N_atom, 1)

    # pair features in dense-trunk windows
    p_lm = (linear(d_lm, p["linear_no_bias_d.weight"]) * v_lm) * mask_trunked[..., None]
    p_lm = p_lm + linear(
        1.0 / (1.0 + (d_lm ** 2).sum(axis=-1, keepdims=True)), p["linear_no_bias_invd.weight"]
    ) * v_lm
    p_lm = p_lm + linear(v_lm.astype(p_lm.dtype), p["linear_no_bias_v.weight"])
    return p_lm, c_l


def _small_mlp(x: mx.array, p: dict) -> mx.array:
    """small_mlp = ReLU -> Linear(1) -> ReLU -> Linear(3) -> ReLU -> Linear(5)."""
    x = mx.maximum(x, 0)
    x = linear(x, p["1.weight"])
    x = mx.maximum(x, 0)
    x = linear(x, p["3.weight"])
    x = mx.maximum(x, 0)
    x = linear(x, p["5.weight"])
    return x


def atom_attention_encoder(
    atom_to_token_idx: mx.array,
    ref_pos: mx.array,
    ref_charge: mx.array,
    ref_mask: mx.array,
    ref_atom_name_chars: mx.array,
    ref_element: mx.array,
    d_lm: mx.array,
    v_lm: mx.array,
    mask_trunked: mx.array,
    p: dict,
    n_blocks: int = 3,
    n_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    ln_eps: float = 1e-5,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """AtomAttentionEncoder forward for has_coords=False. Returns (a, q_l, c_l, p_lm).

    `p` is the input_embedder.atom_attention_encoder subtree.
    """
    p_lm, c_l = atom_attention_encoder_prepare_cache(
        ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars,
        d_lm, v_lm, mask_trunked, p,
    )

    q_l = c_l  # no r_l: q_l = c_l.clone()

    # add atom single context to pair, in windows
    c_l_q, c_l_k, _, _ = rearrange_qk_to_dense_trunk(c_l, c_l, n_queries, n_keys)
    p_lm = (
        p_lm
        + linear(mx.maximum(c_l_q, 0)[..., None, :], p["linear_no_bias_cl.weight"])
        + linear(mx.maximum(c_l_k, 0)[..., None, :, :], p["linear_no_bias_cm.weight"])
    )
    p_lm = p_lm + _small_mlp(p_lm, _sub(p, "small_mlp"))

    # cross-attention atom transformer
    dt = _sub(p, "atom_transformer.diffusion_transformer")
    q_l = atom_transformer(q_l, c_l, p_lm, dt, n_blocks, n_heads, n_queries, n_keys, ln_eps)

    # aggregate atom -> token (mean over relu(linear_q(q_l)))
    a = aggregate_atom_to_token_mean(
        mx.maximum(linear(q_l, p["linear_no_bias_q.weight"]), 0),
        atom_to_token_idx,
        n_token=None,
    )
    return a, q_l, c_l, p_lm


# ---------------------------------------------------------------------------
# AtomAttentionDecoder — transformer.py:2171
# ---------------------------------------------------------------------------
def atom_attention_decoder(
    atom_to_token_idx: mx.array,
    a: mx.array,
    q_skip: mx.array,
    c_skip: mx.array,
    p_skip: mx.array,
    p: dict,
    n_blocks: int = 3,
    n_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    ln_eps: float = 1e-5,
) -> mx.array:
    """AtomAttentionDecoder (AF3 Alg. 6). Returns coordinate update r [..., N_atom, 3].
    `p` is the atom_attention_decoder subtree."""
    q = broadcast_token_to_atom(linear(a, p["linear_no_bias_a.weight"]), atom_to_token_idx) + q_skip
    dt = _sub(p, "atom_transformer.diffusion_transformer")
    q = atom_transformer(q, c_skip, p_skip, dt, n_blocks, n_heads, n_queries, n_keys, ln_eps)
    q = layer_norm(q, p["layernorm_q.weight"], None, ln_eps)
    return linear(q, p["linear_no_bias_out.weight"])
