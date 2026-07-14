"""MLX port of OpenDDE prediction heads.

Ports:
  * ``DistogramHead`` (opendde/model/modules/head.py) — a single biased linear
    over the pair rep, symmetrized ``logits + logits.transpose(-2,-3)``.
  * ``ConfidenceHead`` (opendde/model/modules/confidence.py) — the single-device
    (``OPENDDE_FOLDCP_MODE=single``) path of ``forward`` /
    ``memory_efficient_forward``: builds a z init from the input single rep, adds
    a predicted-distance embedding (one-hot bins + raw distance), runs its own
    4-block Pairformer, then emits pae / pde (pair) and plddt / resolved
    (per-atom, via a token->atom broadcast and a per-tokatom weight gather).

Reuses ``opendde_mlx.modules`` (linear, layer_norm, pairformer_stack) and
``opendde_mlx.weights.remap_pairformer_stack``.  All math is float32.

Real-checkpoint config (verified against opendde.pt, NOT the doc note):
  distogram: c_z=384, no_bins=96, linear HAS bias.
  confidence: c_z=384, c_s=384, c_s_inputs=449, n_blocks=4,
              hidden_scale_up=True  -> tri-att no_heads_pair=12,
              attn_pair_bias n_heads=16, max_atoms_per_token=24,
              b_pae=64, b_pde=64, b_plddt=50, b_resolved=2,
              distance bins arange(3.25, 52.0, 1.25) -> 39 bins.
"""

from __future__ import annotations

import mlx.core as mx

from opendde_mlx import modules as M

# Confidence Pairformer head counts (hidden_scale_up=True, c_z=384).
CONF_NO_HEADS_PAIR_BIAS = 16
CONF_NO_HEADS_PAIR = 12


# ---------------------------------------------------------------------------
# DistogramHead
# ---------------------------------------------------------------------------
def distogram_head(z: mx.array, p: dict) -> mx.array:
    """z: [..., N, N, c_z] -> logits [..., N, N, no_bins], symmetrized.

    p: {"linear.weight": [no_bins, c_z], "linear.bias": [no_bins]}
    """
    logits = M.linear(z, p["linear.weight"], p.get("linear.bias"))
    return logits + mx.swapaxes(logits, -2, -3)


# ---------------------------------------------------------------------------
# helpers for ConfidenceHead
# ---------------------------------------------------------------------------
def _one_hot(x: mx.array, lower: mx.array, upper: mx.array) -> mx.array:
    """Match opendde.model.utils.one_hot:
    dgram = (x[...,None] > lower) * (x[...,None] < upper).float()."""
    xe = x[..., None]
    return (xe > lower).astype(mx.float32) * (xe < upper).astype(mx.float32)


def _cdist(a: mx.array, b: mx.array) -> mx.array:
    """Pairwise euclidean distance, a,b: [..., N, 3] -> [..., N, N] (fp32)."""
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    diff = a[..., :, None, :] - b[..., None, :, :]
    return mx.sqrt(mx.sum(diff * diff, axis=-1))


# ---------------------------------------------------------------------------
# ConfidenceHead
# ---------------------------------------------------------------------------
def confidence_head(
    feats: dict,
    s_inputs: mx.array,
    s_trunk: mx.array,
    z_trunk: mx.array,
    pair_mask: mx.array,
    x_pred_coords: mx.array,
    cp: dict,
    pf: dict,
    n_blocks: int,
    *,
    no_heads_pair_bias: int = CONF_NO_HEADS_PAIR_BIAS,
    no_heads_pair: int = CONF_NO_HEADS_PAIR,
    inf: float = 1e9,
    ln_eps: float = 1e-5,
    compute_plddt: bool = True,
    compute_pae: bool = True,
    compute_pde: bool = True,
    compute_resolved: bool = True,
):
    """Single-device confidence head.

    Args:
        feats: needs "distogram_rep_atom_mask" (bool/0-1, [N_atom]),
               "atom_to_token_idx" ([N_atom]), "atom_to_tokatom_idx" ([N_atom]).
        s_inputs: [N_token, c_s_inputs]
        s_trunk:  [N_token, c_s]
        z_trunk:  [N_token, N_token, c_z]
        pair_mask: [N_token, N_token]
        x_pred_coords: [N_sample, N_atom, 3]
        cp: confidence_head non-pairformer params (flat torch names).
        pf: remapped pairformer-stack p-dict (from remap_pairformer_stack).
        n_blocks: number of pairformer blocks.

    Returns (plddt, pae, pde, resolved), each stacked over N_sample or None.
        plddt:    [N_sample, N_atom, b_plddt]
        pae:      [N_sample, N_token, N_token, b_pae]
        pde:      [N_sample, N_token, N_token, b_pde]
        resolved: [N_sample, N_atom, b_resolved]
    """
    s_trunk = mx.clip(s_trunk, -512.0, 512.0)
    s_trunk = M.layer_norm(
        s_trunk, cp["input_strunk_ln.weight"], cp["input_strunk_ln.bias"], ln_eps
    )

    rep_mask = feats["distogram_rep_atom_mask"]
    rep_idx = mx.array(
        [i for i, v in enumerate(rep_mask.tolist()) if bool(v)], dtype=mx.int32
    )
    # x_pred_coords: [N_sample, N_atom, 3] -> [N_sample, N_token, 3]
    x_rep = mx.take(x_pred_coords, rep_idx, axis=-2)
    n_sample = x_rep.shape[-3]

    lower = cp["lower_bins"]
    upper = cp["upper_bins"]

    # z init from single rep: z_init[i, j] = s2[i] + s1[j]
    s1 = M.linear(s_inputs, cp["linear_no_bias_s1.weight"])
    s2 = M.linear(s_inputs, cp["linear_no_bias_s2.weight"])
    z_base = z_trunk + (s2[..., :, None, :] + s1[..., None, :, :])

    n_token = s_trunk.shape[-2]
    single_mask = mx.ones((n_token,), dtype=s_trunk.dtype)

    atom_to_token_idx = mx.array(feats["atom_to_token_idx"]).astype(mx.int32)
    atom_to_tokatom_idx = mx.array(feats["atom_to_tokatom_idx"]).astype(mx.int32)

    plddt_out = [] if compute_plddt else None
    pae_out = [] if compute_pae else None
    pde_out = [] if compute_pde else None
    resolved_out = [] if compute_resolved else None

    for i in range(n_sample):
        xr = x_rep[..., i, :, :]  # [N_token, 3]
        dist = _cdist(xr, xr)  # [N_token, N_token]

        z = z_base + M.linear(_one_hot(dist, lower, upper), cp["linear_no_bias_d.weight"])
        z = z + M.linear(dist[..., None], cp["linear_no_bias_d_wo_onehot.weight"])

        s_single, z = M.pairformer_stack(
            s_trunk, z, pf, single_mask, pair_mask, n_blocks=n_blocks,
            no_heads_pair_bias=no_heads_pair_bias, no_heads_pair=no_heads_pair,
            inf=inf, ln_eps=ln_eps,
        )

        if compute_pae:
            zln = M.layer_norm(z, cp["pae_ln.weight"], cp["pae_ln.bias"], ln_eps)
            pae_out.append(M.linear(zln, cp["linear_no_bias_pae.weight"]))
        if compute_pde:
            zsym = z + mx.swapaxes(z, -2, -3)
            zln = M.layer_norm(zsym, cp["pde_ln.weight"], cp["pde_ln.bias"], ln_eps)
            pde_out.append(M.linear(zln, cp["linear_no_bias_pde.weight"]))

        if compute_plddt or compute_resolved:
            a = mx.take(s_single, atom_to_token_idx, axis=-2)  # [N_atom, c_s]
            if compute_plddt:
                w = mx.take(cp["plddt_weight"], atom_to_tokatom_idx, axis=0)
                lna = M.layer_norm(a, cp["plddt_ln.weight"], cp["plddt_ln.bias"], ln_eps)
                plddt_out.append(mx.einsum("...nc,ncb->...nb", lna, w))
            if compute_resolved:
                w = mx.take(cp["resolved_weight"], atom_to_tokatom_idx, axis=0)
                lna = M.layer_norm(a, cp["resolved_ln.weight"], cp["resolved_ln.bias"], ln_eps)
                resolved_out.append(mx.einsum("...nc,ncb->...nb", lna, w))

    plddt = mx.stack(plddt_out, axis=-3) if plddt_out is not None else None
    pae = mx.stack(pae_out, axis=-4) if pae_out is not None else None
    pde = mx.stack(pde_out, axis=-4) if pde_out is not None else None
    resolved = mx.stack(resolved_out, axis=-3) if resolved_out is not None else None
    return plddt, pae, pde, resolved
