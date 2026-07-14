"""MLX port of the OpenDDE Boltz-style MSA module.

Mirrors ``opendde/model/modules/pairformer.py``:
  OuterProductMean (opendde/model/triangular/layers.py),
  MSAPairWeightedAveraging, MSAStack, MSABlock, MSAModule.

Boltz per-block order (single / non-Fold-CP path):
    m = msa_stack(m, z)              # MSAPairWeightedAveraging + transition_m
    z = z + outer_product_mean(m)    # write refreshed MSA back into pair rep
    z = pair_stack(z)                # PairformerBlock with c_s=0

Config (real checkpoint): c_m=128, c_z=384, OPM c_hidden=32,
MSAPairWeightedAveraging c=8 n_heads=8, pair_stack hidden_scale_up=True
=> no_heads_pair = c_z // 32 = 12, c_hidden_mul = c_z = 384.

Reuses opendde_mlx.modules helpers (layer_norm, linear, sigmoid,
swiglu_transition, pair_block, _sub).
"""

from __future__ import annotations

import mlx.core as mx

from opendde_mlx.modules import (
    _sub,
    layer_norm,
    linear,
    pair_block,
    sigmoid,
    swiglu_transition,
)


# ---------------------------------------------------------------------------
# OuterProductMean (AF3 Alg. 10)
# ---------------------------------------------------------------------------
def outer_product_mean(
    m: mx.array,
    p: dict,
    mask: mx.array | None = None,
    eps: float = 1e-3,
    ln_eps: float = 1e-5,
) -> mx.array:
    """OuterProductMean.

    m:    [*, N_seq, N_res, c_m]  -> returns [*, N_res, N_res, c_z]

    p keys: layer_norm.{weight,bias}, linear_1.weight, linear_2.weight,
            linear_out.weight, linear_out.bias  (all bias-free except linear_out).
    """
    if mask is None:
        mask = mx.ones(m.shape[:-1], dtype=m.dtype)

    ln = layer_norm(m, p["layer_norm.weight"], p["layer_norm.bias"], ln_eps)

    mask = mask[..., None]                       # [*, N_seq, N_res, 1]
    a = linear(ln, p["linear_1.weight"]) * mask  # [*, N_seq, N_res, C]
    b = linear(ln, p["linear_2.weight"]) * mask

    a = mx.swapaxes(a, -2, -3)                    # [*, N_res, N_seq, C]
    b = mx.swapaxes(b, -2, -3)

    # [*, N_res, N_res, C, C]  (sum over N_seq)
    outer = mx.einsum("...bac,...dae->...bdce", a, b)
    outer = outer.reshape(*outer.shape[:-2], -1)  # [*, N_res, N_res, C*C]

    outer = linear(outer, p["linear_out.weight"], p.get("linear_out.bias"))

    # norm counts contributing sequences per (i, j): [*, N_res, N_res, 1]
    norm = mx.einsum("...abc,...adc->...bdc", mask, mask) + eps
    return outer / norm


# ---------------------------------------------------------------------------
# MSAPairWeightedAveraging (AF3 Alg. 10)
# ---------------------------------------------------------------------------
def msa_pair_weighted_averaging(
    m: mx.array,
    z: mx.array,
    p: dict,
    n_heads: int = 8,
    c: int = 8,
    ln_eps: float = 1e-5,
) -> mx.array:
    """MSAPairWeightedAveraging.

    m: [*, N_seq, N_res, c_m],  z: [*, N_res, N_res, c_z]
    returns update to m: [*, N_seq, N_res, c_m]

    p keys: layernorm_m.{weight,bias}, layernorm_z.{weight,bias},
            linear_no_bias_mv.weight (c*n_heads, c_m),
            linear_no_bias_mg.weight (c*n_heads, c_m),
            linear_no_bias_z.weight  (n_heads, c_z),
            linear_no_bias_out.weight (c_m, c*n_heads).
    """
    mn = layer_norm(m, p["layernorm_m.weight"], p["layernorm_m.bias"], ln_eps)

    v = linear(mn, p["linear_no_bias_mv.weight"])
    v = v.reshape(*v.shape[:-1], n_heads, c)        # [*, N_seq, N_res, H, c]

    g = sigmoid(linear(mn, p["linear_no_bias_mg.weight"]))
    g = g.reshape(*g.shape[:-1], n_heads, c)        # [*, N_seq, N_res, H, c]

    zn = layer_norm(z, p["layernorm_z.weight"], p["layernorm_z.bias"], ln_eps)
    b = linear(zn, p["linear_no_bias_z.weight"])    # [*, N_res, N_res, H]
    w = mx.softmax(b, axis=-2)                       # softmax over j (dim=-2)

    # wv: [*, N_seq, N_res, H, c]  (sum over j)
    wv = mx.einsum("...ijh,...mjhc->...mihc", w, v)
    o = g * wv
    o = o.reshape(*o.shape[:-2], n_heads * c)        # [*, N_seq, N_res, H*c]
    return linear(o, p["linear_no_bias_out.weight"])


# ---------------------------------------------------------------------------
# transition_m remap: Transition (SwiGLU) checkpoint names -> swiglu_transition
# ---------------------------------------------------------------------------
def _remap_transition_m(tp: dict) -> dict:
    """transition_m.* checkpoint keys -> swiglu_transition p-dict."""
    return {
        "layer_norm.weight": tp["layernorm1.weight"],
        "layer_norm.bias": tp["layernorm1.bias"],
        "swiglu.linear_a.weight": tp["linear_no_bias_a.weight"],
        "swiglu.linear_b.weight": tp["linear_no_bias_b.weight"],
        "linear_out.weight": tp["linear_no_bias.weight"],
    }


# ---------------------------------------------------------------------------
# MSAStack: MSAPairWeightedAveraging + transition_m (both residual)
# ---------------------------------------------------------------------------
def msa_stack(m: mx.array, z: mx.array, p: dict, ln_eps: float = 1e-5) -> mx.array:
    """p keys under: msa_pair_weighted_averaging.*, transition_m.*"""
    m = m + msa_pair_weighted_averaging(
        m, z, _sub(p, "msa_pair_weighted_averaging"), ln_eps=ln_eps
    )
    m = m + swiglu_transition(m, _remap_transition_m(_sub(p, "transition_m")), ln_eps=ln_eps)
    return m


# ---------------------------------------------------------------------------
# pair_stack remap: PairformerBlock(c_s=0) checkpoint names -> pair_block p-dict
# ---------------------------------------------------------------------------
def _remap_pair_stack(ps: dict) -> dict:
    """pair_stack.* checkpoint keys -> modules.pair_block p-dict (no prefix)."""
    out = {}
    for k, v in ps.items():
        nk = k
        if k.startswith("tri_att_start.") or k.startswith("tri_att_end."):
            nk = k.replace(".linear.weight", ".linear_z.weight")
        elif k.startswith("pair_transition."):
            tail = k[len("pair_transition."):]
            tail = (tail.replace("layernorm1.", "layer_norm.")
                        .replace("linear_no_bias_a.", "swiglu.linear_a.")
                        .replace("linear_no_bias_b.", "swiglu.linear_b.")
                        .replace("linear_no_bias.", "linear_out."))
            nk = "pair_transition." + tail
        # tri_mul_out.* / tri_mul_in.* names already match pair_block.
        out[nk] = v
    return out


# ---------------------------------------------------------------------------
# MSABlock (Boltz order)
# ---------------------------------------------------------------------------
def msa_block(
    m: mx.array,
    z: mx.array,
    p: dict,
    pair_mask: mx.array,
    is_last: bool = False,
    no_heads_pair: int = 12,
    inf: float = 1e9,
    ln_eps: float = 1e-5,
) -> tuple[mx.array | None, mx.array]:
    """One MSABlock.

    p keys under: msa_stack.*, outer_product_mean_msa.*, pair_stack.*
    Returns (m or None if is_last, z).
    """
    m = msa_stack(m, z, _sub(p, "msa_stack"), ln_eps=ln_eps)
    z = z + outer_product_mean(m, _sub(p, "outer_product_mean_msa"), ln_eps=ln_eps)
    z = pair_block(
        z, _remap_pair_stack(_sub(p, "pair_stack")), pair_mask, no_heads_pair,
        inf=inf, ln_eps=ln_eps,
    )
    if is_last:
        return None, z
    return m, z


# ---------------------------------------------------------------------------
# MSAModule
# ---------------------------------------------------------------------------
def project_msa_sample(msa_sample_feats: mx.array, s_inputs: mx.array, p: dict) -> mx.array:
    """Project raw concatenated MSA features + single inputs into c_m space.

    msa_sample_feats: [*, N_seq, N_res, 34]  (one-hot msa[32] + has_deletion[1]
                      + deletion_value[1], concatenated in that order).
    s_inputs:         [*, N_res, c_s_inputs]

    p keys: linear_no_bias_m.weight (c_m, 34), linear_no_bias_s.weight (c_m, c_s_inputs).
    """
    m = linear(msa_sample_feats, p["linear_no_bias_m.weight"])
    return m + linear(s_inputs, p["linear_no_bias_s.weight"])


def msa_module_blocks(
    m: mx.array,
    z: mx.array,
    p: dict,
    pair_mask: mx.array,
    n_blocks: int = 4,
    no_heads_pair: int = 12,
    inf: float = 1e9,
    ln_eps: float = 1e-5,
) -> mx.array:
    """Run the n MSABlocks over a pre-projected MSA sample m; returns updated z.

    p keys under: blocks.{i}.*
    """
    for i in range(n_blocks):
        is_last = (i + 1 == n_blocks)
        m, z = msa_block(
            m, z, _sub(p, f"blocks.{i}"), pair_mask, is_last=is_last,
            no_heads_pair=no_heads_pair, inf=inf, ln_eps=ln_eps,
        )
    return z
