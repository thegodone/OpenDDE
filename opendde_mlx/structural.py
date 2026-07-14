"""MLX port of StructuralTokenExpander (pair_projection_mode="full") and the
structural_token_refiner (a 4-block PairformerStack with an extra additive
attention bias).

Faithful to ``opendde/model/modules/structural_tokens.py`` (forward path,
pair_chunk_size branch is numerically identical to the full path) and to the
``PairformerBlock`` used by ``opendde/model/modules/pairformer.py`` with
``extra_attn_bias`` threaded into ``AttentionPairBias.standard_multihead_attention``.

Reuses helpers from ``opendde_mlx.modules``.
"""

from __future__ import annotations

import mlx.core as mx

from opendde_mlx.modules import (
    layer_norm,
    linear,
    silu,
    pair_block,
    swiglu_transition,
    mha,
    _sub,
)

# Structural token roles (from opendde.data.tokenizer.STRUCTURAL_TOKEN_ROLES):
#   atom=0 protein_bb=1 protein_sc=2 dna_bb=3 dna_base=4 rna_bb=5 rna_base=6
_BACKBONE_ROLE_IDS = (1, 3, 5)
_SIDECHAIN_ROLE_ID = 2
_BASE_ROLE_IDS = (4, 6)
N_ROLES = 7


# ---------------------------------------------------------------------------
# StructuralTokenExpander (full mode)
# ---------------------------------------------------------------------------
def _emb(table: mx.array, idx: mx.array) -> mx.array:
    """nn.Embedding / index_select: table[idx] over axis 0."""
    return mx.take(table, idx.astype(mx.int32), axis=0)


def _single_split_mlp(s: mx.array, p: dict, ln_eps: float = 1e-5) -> mx.array:
    """Sequential(LayerNorm, LinearNoBias(c_s,2c_s), SiLU, LinearNoBias(2c_s,c_s))."""
    x = layer_norm(s, p["single_split_mlp.0.weight"], p["single_split_mlp.0.bias"], ln_eps)
    x = linear(x, p["single_split_mlp.1.weight"])
    x = silu(x)
    x = linear(x, p["single_split_mlp.3.weight"])
    return x


def _build_pair_features(feats: dict, role: mx.array, parent: mx.array) -> dict:
    """Reproduces _build_structural_pair_context + _build_structural_pair_features_for_rows
    with row_index = arange(n_struct) (the full forward path)."""
    n_struct = role.shape[-1]

    asym_id = _emb(feats["asym_id"], parent)  # [N]

    def role_in(r, ids):
        m = r == ids[0]
        for v in ids[1:]:
            m = m | (r == v)
        return m

    is_backbone = role_in(role, _BACKBONE_ROLE_IDS)
    is_sidechain = role == _SIDECHAIN_ROLE_ID
    is_base = role_in(role, _BASE_ROLE_IDS)

    if "prev_parent_residue_idx" in feats:
        prev_parent = feats["prev_parent_residue_idx"].astype(mx.int32)
    else:
        prev_parent = mx.full((n_struct,), -1, dtype=mx.int32)
    if "next_parent_residue_idx" in feats:
        next_parent = feats["next_parent_residue_idx"].astype(mx.int32)
    else:
        next_parent = mx.full((n_struct,), -1, dtype=mx.int32)

    pi = parent[:, None]
    pj = parent[None, :]
    same_parent_residue = pi == pj
    same_chain = asym_id[:, None] == asym_id[None, :]

    ib_i = is_backbone[:, None]
    ib_j = is_backbone[None, :]
    isc_i = is_sidechain[:, None]
    isc_j = is_sidechain[None, :]
    iba_i = is_base[:, None]
    iba_j = is_base[None, :]

    same_residue_twin = same_parent_residue & (
        (ib_i & (isc_j | iba_j)) | (ib_j & (isc_i | iba_i))
    )
    prev_bb_chain = ib_i & ib_j & same_chain & (prev_parent[:, None] == pj)
    next_bb_chain = ib_i & ib_j & same_chain & (next_parent[:, None] == pj)

    # role_pair_type: default 7, then override in the same order as torch.
    role_pair_type = mx.full((n_struct, n_struct), 7, dtype=mx.int32)

    def setp(rpt, mask, val):
        return mx.where(mask, mx.array(val, dtype=mx.int32), rpt)

    role_pair_type = setp(role_pair_type, ib_i & ib_j, 0)
    role_pair_type = setp(role_pair_type, ib_i & isc_j, 1)
    role_pair_type = setp(role_pair_type, isc_i & ib_j, 2)
    role_pair_type = setp(role_pair_type, isc_i & isc_j, 3)
    role_pair_type = setp(role_pair_type, ib_i & iba_j, 4)
    role_pair_type = setp(role_pair_type, iba_i & ib_j, 5)
    role_pair_type = setp(role_pair_type, iba_i & iba_j, 6)

    return {
        "same_parent_residue": same_parent_residue,
        "same_residue_twin": same_residue_twin,
        "prev_bb_chain": prev_bb_chain,
        "next_bb_chain": next_bb_chain,
        "role_pair_type": role_pair_type,
    }


def _pair_project_by_role_full(z: mx.array, role: mx.array, p: dict) -> mx.array:
    """_pair_project_by_role_full: per (role_i, role_j) LinearNoBias applied to the
    masked entries. Masks partition all (i, j) since roles are in [0, n_roles)."""
    n = role.shape[-1]
    role_i = role[:, None]
    role_j = role[None, :]
    delta = mx.zeros_like(z)
    for ri in range(N_ROLES):
        for rj in range(N_ROLES):
            mask = (role_i == ri) & (role_j == rj)  # [N, N]
            if not bool(mx.any(mask).item()):
                continue
            w = p[f"pair_block_proj.{ri * N_ROLES + rj}.weight"]
            proj = z @ w.T
            delta = mx.where(mask[..., None], proj, delta)
    return delta


def _make_pair_init_bias(pf: dict, p: dict) -> mx.array:
    b = _emb(p["same_parent_embedding.weight"], pf["same_parent_residue"].astype(mx.int32))
    b = b + _emb(p["same_residue_twin_embedding.weight"], pf["same_residue_twin"].astype(mx.int32))
    b = b + _emb(p["prev_bb_chain_embedding.weight"], pf["prev_bb_chain"].astype(mx.int32))
    b = b + _emb(p["next_bb_chain_embedding.weight"], pf["next_bb_chain"].astype(mx.int32))
    b = b + _emb(p["role_pair_type_embedding.weight"], pf["role_pair_type"])
    return b


def _make_attention_bias(pf: dict, p: dict) -> mx.array:
    role_pair_bias = _emb(p["attn_bias_role_pair_type"][:, None], pf["role_pair_type"])[..., 0]
    return (
        p["attn_bias_same_parent"] * pf["same_parent_residue"].astype(mx.float32)
        + p["attn_bias_same_residue_twin"] * pf["same_residue_twin"].astype(mx.float32)
        + p["attn_bias_prev_bb_chain"] * pf["prev_bb_chain"].astype(mx.float32)
        + p["attn_bias_next_bb_chain"] * pf["next_bb_chain"].astype(mx.float32)
        + role_pair_bias
    )


def structural_token_expander(
    p: dict,
    feats: dict,
    s_inputs_res: mx.array,
    s_res: mx.array,
    z_res: mx.array,
    ln_eps: float = 1e-5,
):
    """MLX port of StructuralTokenExpander.forward (full mode).

    ``p``     : flat expander subtree (mlx arrays), keys as in the checkpoint.
    ``feats`` : dict of int mlx arrays: parent_residue_idx, subtoken_role_id,
                asym_id, (optional) prev_parent_residue_idx, next_parent_residue_idx.
    Returns (s_inputs_struct, s_struct, z_struct, pair_features) where
    pair_features["structural_pair_attn_bias"] is the [N, N] attention bias.
    """
    parent = feats["parent_residue_idx"].astype(mx.int32)
    role = feats["subtoken_role_id"].astype(mx.int32)

    s_inputs_struct = _emb(s_inputs_res, parent) + _emb(p["single_input_role_embedding.weight"], role)
    s_parent = _emb(s_res, parent)
    s_struct = s_parent + _single_split_mlp(s_parent, p, ln_eps) + _emb(p["single_role_embedding.weight"], role)

    pf = _build_pair_features(feats, role, parent)

    # z_parent = z_res[parent][:, parent]
    z_parent = mx.take(mx.take(z_res, parent, axis=-3), parent, axis=-2)
    z_struct = z_parent + _pair_project_by_role_full(z_parent, role, p)
    z_struct = z_struct + _make_pair_init_bias(pf, p)

    pf["structural_pair_attn_bias"] = _make_attention_bias(pf, p)
    return s_inputs_struct, s_struct, z_struct, pf


# ---------------------------------------------------------------------------
# structural_token_refiner (PairformerStack, has_s=False, + extra_attn_bias)
# ---------------------------------------------------------------------------
def _attn_pair_bias_extra(
    a: mx.array,
    z: mx.array,
    p: dict,
    no_heads: int,
    extra_attn_bias: mx.array | None,
    ln_eps: float = 1e-5,
) -> mx.array:
    """AttentionPairBias (has_s=False) with an additive extra bias, matching
    ``standard_multihead_attention``: bias = permute(linear_z(ln_z(z))) [+ extra]."""
    an = layer_norm(a, p["layer_norm_a.weight"], p["layer_norm_a.bias"], ln_eps)

    zb = layer_norm(z, p["layer_norm_z.weight"], p["layer_norm_z.bias"], ln_eps)
    zb = linear(zb, p["linear_z.weight"])  # [*, N, N, H]
    zb = mx.moveaxis(zb, -1, -3)  # -> [*, H, N, N]
    biases = [zb]
    if extra_attn_bias is not None:
        # extra is [N, N]; broadcasts across the head axis of [*, H, N, N] scores.
        biases.append(extra_attn_bias)

    return mha(an, an, _sub(p, "mha"), no_heads, biases=tuple(biases), gating=True)


def _refiner_block(
    s: mx.array,
    z: mx.array,
    p: dict,
    pair_mask: mx.array,
    no_heads_pair_bias: int,
    no_heads_pair: int,
    extra_attn_bias: mx.array | None,
    inf: float = 1e9,
    ln_eps: float = 1e-5,
):
    z = pair_block(z, _sub(p, "pair_stack"), pair_mask, no_heads_pair, inf, ln_eps)
    s = s + _attn_pair_bias_extra(
        s, z, _sub(p, "attn_pair_bias"), no_heads_pair_bias, extra_attn_bias, ln_eps
    )
    s = s + swiglu_transition(s, _sub(p, "single_transition"), mask=None, ln_eps=ln_eps)
    return s, z


def structural_token_refiner(
    s: mx.array,
    z: mx.array,
    p: dict,
    n_blocks: int,
    extra_attn_bias: mx.array | None = None,
    no_heads_pair_bias: int = 8,
    no_heads_pair: int = 12,
    inf: float = 1e9,
    ln_eps: float = 1e-5,
):
    """4-block PairformerStack refiner. ``p`` is the remapped stack p-dict
    (blocks.{i}.*). pair_mask=None (mirrors opendde.py call). ``no_heads_pair``=12
    comes from hidden_scale_up (c_z // c_hidden_pair_att = 384 // 32)."""
    n = s.shape[-2]
    pair_mask = mx.ones((*z.shape[:-3], n, n), dtype=z.dtype)
    for i in range(n_blocks):
        s, z = _refiner_block(
            s, z, _sub(p, f"blocks.{i}"), pair_mask,
            no_heads_pair_bias, no_heads_pair, extra_attn_bias, inf, ln_eps,
        )
    return s, z
