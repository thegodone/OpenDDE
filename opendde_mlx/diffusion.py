"""MLX port of the OpenDDE DiffusionModule (AF3 Algorithm 20) and its
DiffusionConditioning (Algorithm 21), plus the supporting DiffusionTransformer
(Alg 23/24/25), AtomAttentionEncoder (Alg 5), AtomAttentionDecoder (Alg 6) and
AtomTransformer (Alg 7).

Only the single-process (non Fold-CP) inference path is ported.  The active
diffusion-pair compression (c_z 384 -> c_z_pair_diffusion 128) is included.

Weight p-dicts keep the torch submodule key names verbatim (use
``weights.subtree`` / ``modules._sub`` to slice the checkpoint), so the real
checkpoint loads 1:1 without renaming.
"""

from __future__ import annotations

import math

import mlx.core as mx

from opendde_mlx.modules import linear, layer_norm, sigmoid, silu, _sub
from opendde_mlx.atom import rearrange_qk_to_dense_trunk, local_attention


# ---------------------------------------------------------------------------
# small building blocks
# ---------------------------------------------------------------------------
def adaptive_layer_norm(a: mx.array, s: mx.array, p: dict, eps: float = 1e-5) -> mx.array:
    """AdaptiveLayerNorm (AF3 Alg 26).

    a_norm = LN(a) (no scale/offset); s_norm = LN(s) (scale only)
    out = sigmoid(linear_s(s_norm)) * a_norm + linear_nobias_s(s_norm)

    p keys: layernorm_s.weight, linear_s.weight, linear_s.bias, linear_nobias_s.weight
    """
    a_n = layer_norm(a, None, None, eps)
    s_n = layer_norm(s, p["layernorm_s.weight"], None, eps)
    scale = sigmoid(linear(s_n, p["linear_s.weight"], p["linear_s.bias"]))
    return scale * a_n + linear(s_n, p["linear_nobias_s.weight"])


def transition(x: mx.array, p: dict, eps: float = 1e-5) -> mx.array:
    """primitives.Transition (SwiGLU-style, Alg 11).

    p keys: layernorm1.{weight,bias}, linear_no_bias_a.weight,
            linear_no_bias_b.weight, linear_no_bias.weight
    """
    y = layer_norm(x, p["layernorm1.weight"], p["layernorm1.bias"], eps)
    a = silu(linear(y, p["linear_no_bias_a.weight"]))
    b = linear(y, p["linear_no_bias_b.weight"])
    return linear(a * b, p["linear_no_bias.weight"])


def _split_heads(t: mx.array, n_heads: int) -> mx.array:
    """[*, N, H*ch] -> [*, H, N, ch]"""
    t = t.reshape(*t.shape[:-1], n_heads, -1)
    return mx.swapaxes(t, -2, -3)


def conditioned_transition_block(
    a: mx.array, s: mx.array, p: dict, eps: float = 1e-5
) -> mx.array:
    """ConditionedTransitionBlock (Alg 25).

    p keys: adaln.*, linear_nobias_a1.weight, linear_nobias_a2.weight,
            linear_nobias_b.weight, linear_s.{weight,bias}
    """
    a_n = adaptive_layer_norm(a, s, _sub(p, "adaln"), eps)
    b = silu(linear(a_n, p["linear_nobias_a1.weight"])) * linear(
        a_n, p["linear_nobias_a2.weight"]
    )
    gate = sigmoid(linear(s, p["linear_s.weight"], p["linear_s.bias"]))
    return gate * linear(b, p["linear_nobias_b.weight"])


# ---------------------------------------------------------------------------
# AttentionPairBias with AdaLN (AF3 Alg 24) — standard + local variants
# ---------------------------------------------------------------------------
def adaln_attention_pair_bias(
    a: mx.array,
    s: mx.array,
    z: mx.array,
    p: dict,
    n_heads: int,
    cross: bool = False,
    n_queries: int | None = None,
    n_keys: int | None = None,
    inf: float = 1e10,
    eps: float = 1e-5,
    extra_attn_bias: mx.array | None = None,
) -> mx.array:
    """AttentionPairBias.forward with has_s=True.

    p keys: layernorm_a.*, (layernorm_kv.* if cross), layernorm_z.weight,
            linear_nobias_z.weight, attention.linear_{q,k,v,g,o}.*,
            linear_a_last.{weight,bias}
    """
    a_n = adaptive_layer_norm(a, s, _sub(p, "layernorm_a"), eps)
    if cross:
        kv = adaptive_layer_norm(a_n, s, _sub(p, "layernorm_kv"), eps)
    else:
        kv = a_n

    pa = _sub(p, "attention")
    q = _split_heads(linear(a_n, pa["linear_q.weight"], pa.get("linear_q.bias")), n_heads)
    k = _split_heads(linear(kv, pa["linear_k.weight"], pa.get("linear_k.bias")), n_heads)
    v = _split_heads(linear(kv, pa["linear_v.weight"], pa.get("linear_v.bias")), n_heads)
    c_hidden = q.shape[-1]
    scale = 1.0 / math.sqrt(c_hidden)
    q = q * scale

    zb = layer_norm(z, p["layernorm_z.weight"], None, eps)
    zb = linear(zb, p["linear_nobias_z.weight"])  # [..., ?, ?, H]

    if n_queries and n_keys:
        # local windowed attention. z: [..., n_blocks, n_q, n_k, H]
        # permute_final_dims [3,0,1,2] -> [..., H, n_blocks, n_q, n_k]
        bias = mx.moveaxis(zb, -1, -4)
        o = local_attention(q, k, v, n_queries, n_keys, trunked_attn_bias=bias, inf=inf)
    else:
        # full attention. z: [..., N, N, H] -> [..., H, N, N]
        bias = mx.moveaxis(zb, -1, -3)
        if extra_attn_bias is not None:
            # structural_pair_attn_bias [N, N] -> broadcast over heads
            bias = bias + extra_attn_bias
        qs = q  # already scaled
        scores = mx.einsum("...qc,...kc->...qk", qs, k) + bias
        attn = mx.softmax(scores, axis=-1)
        o = mx.einsum("...qk,...kc->...qc", attn, v)  # [..., H, N, ch]

    o = mx.swapaxes(o, -2, -3)  # [..., N, H, ch]
    g = sigmoid(linear(a_n, pa["linear_g.weight"], pa.get("linear_g.bias")))
    g = g.reshape(*g.shape[:-1], n_heads, -1)
    o = o * g
    o = o.reshape(*o.shape[:-2], -1)
    o = linear(o, pa["linear_o.weight"])

    gate = sigmoid(linear(s, p["linear_a_last.weight"], p["linear_a_last.bias"]))
    return gate * o


def diffusion_transformer_block(
    a: mx.array,
    s: mx.array,
    z: mx.array,
    p: dict,
    n_heads: int,
    cross: bool = False,
    n_queries: int | None = None,
    n_keys: int | None = None,
    inf: float = 1e10,
    eps: float = 1e-5,
    extra_attn_bias: mx.array | None = None,
) -> mx.array:
    """One DiffusionTransformerBlock (Alg 23 line 2-3)."""
    attn = adaln_attention_pair_bias(
        a, s, z, _sub(p, "attention_pair_bias"), n_heads,
        cross=cross, n_queries=n_queries, n_keys=n_keys, inf=inf, eps=eps,
        extra_attn_bias=extra_attn_bias,
    )
    a = attn + a
    ff = conditioned_transition_block(a, s, _sub(p, "conditioned_transition_block"), eps)
    return ff + a


def diffusion_transformer(
    a: mx.array,
    s: mx.array,
    z: mx.array,
    p: dict,
    n_blocks: int,
    n_heads: int,
    cross: bool = False,
    n_queries: int | None = None,
    n_keys: int | None = None,
    inf: float = 1e10,
    eps: float = 1e-5,
    extra_attn_bias: mx.array | None = None,
) -> mx.array:
    """DiffusionTransformer (Alg 23), p keys: blocks.{i}.*"""
    for i in range(n_blocks):
        a = diffusion_transformer_block(
            a, s, z, _sub(p, f"blocks.{i}"), n_heads,
            cross=cross, n_queries=n_queries, n_keys=n_keys, inf=inf, eps=eps,
            extra_attn_bias=extra_attn_bias,
        )
    return a


def atom_transformer(
    q: mx.array,
    c: mx.array,
    pmat: mx.array,
    p: dict,
    n_blocks: int = 3,
    n_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    inf: float = 1e10,
    eps: float = 1e-5,
) -> mx.array:
    """AtomTransformer (Alg 7): cross-attention local diffusion transformer.

    p keys: diffusion_transformer.blocks.{i}.*
    """
    return diffusion_transformer(
        q, c, pmat, _sub(p, "diffusion_transformer"), n_blocks, n_heads,
        cross=True, n_queries=n_queries, n_keys=n_keys, inf=inf, eps=eps,
    )


# ---------------------------------------------------------------------------
# gather / broadcast / aggregate helpers
# ---------------------------------------------------------------------------
def gather_pair_in_dense_trunk(
    z: mx.array, idx_q: mx.array, idx_k: mx.array
) -> mx.array:
    """y[..., b, i, j, :] = z[..., idx_q[b,i], idx_k[b,j], :].

    z: [..., N, N, c]; idx_q: [N_b, N_q]; idx_k: [N_b, N_k]
    -> [..., N_b, N_q, N_k, c]
    """
    N = z.shape[-2]
    c = z.shape[-1]
    N_b, N_q = idx_q.shape
    N_k = idx_k.shape[1]
    idx_q_e = mx.broadcast_to(idx_q[:, :, None], (N_b, N_q, N_k))
    idx_k_e = mx.broadcast_to(idx_k[:, None, :], (N_b, N_q, N_k))
    flat_idx = (idx_q_e * N + idx_k_e).reshape(-1)  # [N_b*N_q*N_k]
    z_flat = z.reshape(*z.shape[:-3], N * N, c)
    y = mx.take(z_flat, flat_idx, axis=-2)  # [..., N_b*N_q*N_k, c]
    return y.reshape(*z.shape[:-3], N_b, N_q, N_k, c)


def broadcast_token_to_atom(x_token: mx.array, atom_to_token_idx: mx.array) -> mx.array:
    """x_token: [..., N_token, d]; idx: [N_atom] -> [..., N_atom, d]"""
    return mx.take(x_token, atom_to_token_idx, axis=-2)


def aggregate_atom_to_token_mean(
    x_atom: mx.array, atom_to_token_idx: mx.array, n_token: int
) -> mx.array:
    """Mean-pool atoms into tokens. x_atom: [..., N_atom, d] -> [..., N_token, d].

    Matches scatter_mean (sum / count, count clamped to >=1).
    """
    N_atom = x_atom.shape[-2]
    idx = atom_to_token_idx.astype(mx.int32)
    onehot = (idx[None, :] == mx.arange(n_token)[:, None]).astype(x_atom.dtype)  # [T, A]
    summed = mx.einsum("ta,...ad->...td", onehot, x_atom)
    count = mx.sum(onehot, axis=-1)  # [T]
    count = mx.maximum(count, mx.array(1.0, x_atom.dtype))
    return summed / count[:, None]


# ---------------------------------------------------------------------------
# AtomAttentionEncoder (Alg 5) — has_coords path
# ---------------------------------------------------------------------------
def atom_attention_encoder(
    p: dict,
    atom_to_token_idx: mx.array,
    ref_pos: mx.array,
    ref_charge: mx.array,
    ref_mask: mx.array,
    ref_atom_name_chars: mx.array,
    ref_element: mx.array,
    d_lm: mx.array,
    v_lm: mx.array,
    mask_trunked: mx.array,
    r_l: mx.array,
    s: mx.array,
    z: mx.array,
    n_queries: int = 32,
    n_keys: int = 128,
    n_blocks: int = 3,
    n_heads: int = 4,
    eps: float = 1e-5,
):
    """Returns (a, q_skip, c_skip, p_skip).

    d_lm: [..., n_blocks, n_q, n_k, 3]; v_lm: [..., n_blocks, n_q, n_k, 1];
    mask_trunked: [n_blocks, n_q, n_k] (0/1);
    r_l: [..., N_sample, N_atom, 3]; s: [..., N_sample, N_token, c_s];
    z: [..., N_token, N_token, c_z_pair_diffusion].
    """
    batch = ref_pos.shape[:-2]
    n_atom = ref_pos.shape[-2]
    n_token = s.shape[-2]

    # --- prepare_cache: atom single c_l ---
    c_l = linear(ref_pos, p["linear_no_bias_ref_pos.weight"]) + linear(
        mx.arcsinh(ref_charge).reshape(*batch, n_atom, 1),
        p["linear_no_bias_ref_charge.weight"],
    )
    ref_features = mx.concatenate(
        [
            ref_mask.reshape(*batch, n_atom, 1),
            ref_element.reshape(*batch, n_atom, 128),
            ref_atom_name_chars.reshape(*batch, n_atom, 4 * 64),
        ],
        axis=-1,
    )
    c_l = c_l + linear(ref_features, p["linear_no_bias_f.weight"])
    c_l = c_l * ref_mask.reshape(*batch, n_atom, 1)

    # --- prepare_cache: atom pair p_lm ---
    mt = mask_trunked[..., None]
    p_lm = (linear(d_lm, p["linear_no_bias_d.weight"]) * v_lm) * mt
    invd = 1.0 / (1.0 + mx.sum(d_lm ** 2, axis=-1, keepdims=True))
    p_lm = p_lm + linear(invd, p["linear_no_bias_invd.weight"]) * v_lm
    p_lm = p_lm + linear(v_lm, p["linear_no_bias_v.weight"])

    # add token-pair trunk context (has_coords)
    idx_q, idx_k, _, _ = rearrange_qk_to_dense_trunk(
        atom_to_token_idx[:, None], atom_to_token_idx[:, None], n_queries, n_keys
    )
    idx_q = idx_q[..., 0].astype(mx.int32)  # [n_blocks, n_q]
    idx_k = idx_k[..., 0].astype(mx.int32)  # [n_blocks, n_k]
    z_tok = gather_pair_in_dense_trunk(z, idx_q, idx_k)  # [..., n_blocks, n_q, n_k, c_z]
    z_tok = linear(layer_norm(z_tok, p["layernorm_z.weight"], None, eps),
                   p["linear_no_bias_z.weight"])
    # p_lm gains a broadcast sample dim (torch unsqueeze(-5))
    p_lm = p_lm[..., None, :, :, :, :] + z_tok[..., None, :, :, :, :]

    # --- single conditioning broadcast + noisy positions ---
    s_proj = linear(layer_norm(s, p["layernorm_s.weight"], None, eps),
                    p["linear_no_bias_s.weight"])  # [..., N_sample, N_token, c_atom]
    c_l = c_l[..., None, :, :] + broadcast_token_to_atom(s_proj, atom_to_token_idx)
    q_l = c_l + linear(r_l, p["linear_no_bias_r.weight"])  # [..., N_sample, N_atom, c_atom]

    # --- add atom single context to pair + small MLP ---
    c_l_q, c_l_k, _, _ = rearrange_qk_to_dense_trunk(c_l, c_l, n_queries, n_keys)
    p_chunk = (
        p_lm
        + linear(mx.maximum(c_l_q[..., None, :], 0.0), p["linear_no_bias_cl.weight"])
        + linear(mx.maximum(c_l_k[..., None, :, :], 0.0), p["linear_no_bias_cm.weight"])
    )
    mlp = mx.maximum(p_chunk, 0.0)
    mlp = mx.maximum(linear(mlp, p["small_mlp.1.weight"]), 0.0)
    mlp = mx.maximum(linear(mlp, p["small_mlp.3.weight"]), 0.0)
    mlp = linear(mlp, p["small_mlp.5.weight"])
    p_lm = p_chunk + mlp

    # --- atom transformer ---
    q_l = atom_transformer(
        q_l, c_l, p_lm, _sub(p, "atom_transformer"),
        n_blocks=n_blocks, n_heads=n_heads, n_queries=n_queries, n_keys=n_keys, eps=eps,
    )

    # --- aggregate to tokens ---
    a = aggregate_atom_to_token_mean(
        mx.maximum(linear(q_l, p["linear_no_bias_q.weight"]), 0.0),
        atom_to_token_idx, n_token,
    )
    return a, q_l, c_l, p_lm


# ---------------------------------------------------------------------------
# AtomAttentionDecoder (Alg 6)
# ---------------------------------------------------------------------------
def atom_attention_decoder(
    p: dict,
    atom_to_token_idx: mx.array,
    a: mx.array,
    q_skip: mx.array,
    c_skip: mx.array,
    p_skip: mx.array,
    n_queries: int = 32,
    n_keys: int = 128,
    n_blocks: int = 3,
    n_heads: int = 4,
    eps: float = 1e-5,
) -> mx.array:
    q = broadcast_token_to_atom(linear(a, p["linear_no_bias_a.weight"]), atom_to_token_idx) + q_skip
    q = atom_transformer(
        q, c_skip, p_skip, _sub(p, "atom_transformer"),
        n_blocks=n_blocks, n_heads=n_heads, n_queries=n_queries, n_keys=n_keys, eps=eps,
    )
    q = layer_norm(q, p["layernorm_q.weight"], None, eps)
    return linear(q, p["linear_no_bias_out.weight"])


# ---------------------------------------------------------------------------
# DiffusionConditioning (Alg 21) — with active pair compression
# ---------------------------------------------------------------------------
def fourier_embedding(t: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """cos(2 pi (t[..., None] * w + b)). t: [..., N_sample] -> [..., N_sample, c]"""
    return mx.cos(2.0 * math.pi * (t[..., None] * w + b))


def _apply_pair_transitions(pair_z: mx.array, p: dict, eps: float = 1e-5) -> mx.array:
    pair_z = pair_z + transition(pair_z, _sub(p, "transition_z1"), eps)
    pair_z = pair_z + transition(pair_z, _sub(p, "transition_z2"), eps)
    return pair_z


def diffusion_conditioning(
    p: dict,
    t_hat: mx.array,
    relp_feature: mx.array,
    s_inputs: mx.array,
    s_trunk: mx.array,
    z_trunk: mx.array,
    sigma_data: float = 16.0,
    eps: float = 1e-5,
):
    """Returns (single_s, pair_z).

    single_s: [..., N_sample, N_token, c_s]; pair_z: [..., N_token, N_token, c_z_pair_diffusion]
    relp_feature: [..., N_token, N_token, 139]; t_hat: [..., N_sample]
    """
    # --- pair conditioning (prepare_cache) with active compression ---
    z_pair_trunk = linear(
        layer_norm(z_trunk, p["layernorm_z_trunk.weight"], None, eps),
        p["linear_no_bias_z_trunk.weight"],
    )  # 384 -> 128
    relpe = linear(relp_feature, p["relpe.linear_no_bias.weight"])  # 139 -> 128
    pair_z = mx.concatenate([z_pair_trunk, relpe], axis=-1)  # [..., N, N, 256]
    pair_z = linear(layer_norm(pair_z, p["layernorm_z.weight"], None, eps),
                    p["linear_no_bias_z.weight"])  # 256 -> 128
    pair_z = _apply_pair_transitions(pair_z, p, eps)

    # --- single conditioning ---
    single_s = mx.concatenate([s_trunk, s_inputs], axis=-1)  # [..., N, c_s + c_s_inputs]
    single_s = linear(layer_norm(single_s, p["layernorm_s.weight"], None, eps),
                      p["linear_no_bias_s.weight"])  # -> c_s

    noise_ratio = mx.maximum(t_hat / sigma_data, 1e-10)
    noise_n = fourier_embedding(
        mx.log(noise_ratio) / 4.0,
        p["fourier_embedding.w"], p["fourier_embedding.b"],
    )
    noise_proj = linear(layer_norm(noise_n, p["layernorm_n.weight"], None, eps),
                        p["linear_no_bias_n.weight"])  # [..., N_sample, c_s]
    single_s = single_s[..., None, :, :] + noise_proj[..., None, :]  # [..., N_sample, N, c_s]

    single_s = single_s + transition(single_s, _sub(p, "transition_s1"), eps)
    single_s = single_s + transition(single_s, _sub(p, "transition_s2"), eps)
    return single_s, pair_z


# ---------------------------------------------------------------------------
# DiffusionModule (Alg 20)
# ---------------------------------------------------------------------------
def diffusion_module_forward(
    p: dict,
    x_noisy: mx.array,
    t_hat: mx.array,
    input_feature_dict: dict,
    s_inputs: mx.array,
    s_trunk: mx.array,
    z_trunk: mx.array,
    sigma_data: float = 16.0,
    eps: float = 1e-5,
) -> mx.array:
    """One denoise step: (x_noisy, t_hat, s_inputs, s_trunk, z_trunk) -> x_denoised.

    input_feature_dict must contain: relp, atom_to_token_idx, ref_pos, ref_charge,
    ref_mask, ref_atom_name_chars, ref_element, d_lm, v_lm, mask_trunked.

    x_noisy: [..., N_sample, N_atom, 3]; t_hat: [..., N_sample]
    """
    # EDM r_noisy = c_in * x
    r_noisy = x_noisy / mx.sqrt(sigma_data ** 2 + t_hat ** 2)[..., None, None]

    # conditioning
    s_single, z_pair = diffusion_conditioning(
        _sub(p, "diffusion_conditioning"),
        t_hat, input_feature_dict["relp"], s_inputs, s_trunk, z_trunk,
        sigma_data=sigma_data, eps=eps,
    )

    # atom encoder needs s_trunk broadcast with a sample dim (expand_at_dim -3, n=1)
    s_enc = s_trunk[..., None, :, :]

    a_token, q_skip, c_skip, p_skip = atom_attention_encoder(
        _sub(p, "atom_attention_encoder"),
        input_feature_dict["atom_to_token_idx"],
        input_feature_dict["ref_pos"],
        input_feature_dict["ref_charge"],
        input_feature_dict["ref_mask"],
        input_feature_dict["ref_atom_name_chars"],
        input_feature_dict["ref_element"],
        input_feature_dict["d_lm"],
        input_feature_dict["v_lm"],
        input_feature_dict["mask_trunked"],
        r_noisy, s_enc, z_pair, eps=eps,
    )

    a_token = a_token + linear(
        layer_norm(s_single, p["layernorm_s.weight"], None, eps),
        p["linear_no_bias_s.weight"],
    )  # [..., N_sample, N_token, c_token]

    a_token = diffusion_transformer(
        a_token, s_single, z_pair, _sub(p, "diffusion_transformer"),
        n_blocks=24, n_heads=16, cross=False, eps=eps,
        extra_attn_bias=input_feature_dict.get("structural_pair_attn_bias"),
    )
    a_token = layer_norm(a_token, p["layernorm_a.weight"], None, eps)

    r_update = atom_attention_decoder(
        _sub(p, "atom_attention_decoder"),
        input_feature_dict["atom_to_token_idx"],
        a_token, q_skip, c_skip, p_skip, eps=eps,
    )

    # EDM: D = c_skip * x + c_out * r_update
    s_ratio = (t_hat / sigma_data)[..., None, None]
    x_denoised = (
        1.0 / (1.0 + s_ratio ** 2) * x_noisy
        + t_hat[..., None, None] / mx.sqrt(1.0 + s_ratio ** 2) * r_update
    )
    return x_denoised
