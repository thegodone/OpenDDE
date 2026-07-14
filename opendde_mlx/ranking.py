# SPDX-License-Identifier: Apache-2.0
# MLX port of opendde ranking / clash scoring (opendde/model/sample_confidence.py
# logits_to_score + calculate_ptm/iptm + ranking_score, and opendde/metrics/clash.py
# af3 clash detection). No learned parameters: this is pure math over confidence
# logits + predicted coordinates. ptm/iptm/logits_to_score run in MLX; the af3 clash
# graph search runs host-side in numpy (as permitted by the task).

from __future__ import annotations

import mlx.core as mx
import numpy as np

_INF = float("inf")


# ----------------------------------------------------------------------------- bins
def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> mx.array:
    """Mirror torch: linspace(min, max-bw, no_bins) + 0.5*bw."""
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = mx.linspace(min_bin, max_bin - bin_width, no_bins)
    return boundaries + 0.5 * bin_width


def logits_to_prob(logits: mx.array, axis: int = -1) -> mx.array:
    return mx.softmax(logits, axis=axis)


def logits_to_score(
    logits: mx.array,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    return_prob: bool = False,
):
    """Convert logits [..., no_bins] -> expected value over bin centers [...]."""
    prob = logits_to_prob(logits, axis=-1)
    bin_centers = get_bin_centers(min_bin, max_bin, no_bins).astype(prob.dtype)
    score = prob @ bin_centers
    if return_prob:
        return score, prob
    return score


def calculate_normalization(N: int) -> float:
    """TM-score normalization constant d0(N)."""
    return 1.24 * (max(N, 19) - 15) ** (1.0 / 3.0) - 1.8


# ------------------------------------------------------------------------- ptm/iptm
def _select_token_mask(pae_prob, has_frame, token_mask, asym_id=None):
    """Apply a boolean token_mask (numpy/bool) to pae_prob [...,Nt,Nt,B] on both
    token axes plus has_frame / asym_id vectors."""
    idx = mx.array(np.nonzero(np.asarray(token_mask, dtype=bool))[0].astype(np.int32))
    pae_prob = mx.take(pae_prob, idx, axis=-3)
    pae_prob = mx.take(pae_prob, idx, axis=-2)
    hf = has_frame[idx]
    if asym_id is not None:
        return pae_prob, hf, asym_id[idx]
    return pae_prob, hf


def _masked_max_over_frames(per_token: mx.array, has_frame_np: np.ndarray) -> mx.array:
    """max over the last axis, restricted to has_frame positions."""
    hf = mx.array(has_frame_np.astype(np.float32))  # [N_d]
    masked = mx.where(hf > 0, per_token, mx.array(-_INF, dtype=per_token.dtype))
    return mx.max(masked, axis=-1)


def calculate_ptm(pae_prob, has_frame, min_bin, max_bin, no_bins, token_mask=None):
    """pTM. pae_prob [..., Nt, Nt, B]; has_frame bool [Nt]. Returns [...]."""
    has_frame_np = np.asarray(has_frame, dtype=bool)
    if token_mask is not None:
        tm = np.asarray(token_mask, dtype=bool)
        idx = np.nonzero(tm)[0].astype(np.int32)
        pae_prob = mx.take(mx.take(pae_prob, mx.array(idx), axis=-3), mx.array(idx), axis=-2)
        has_frame_np = has_frame_np[idx]

    if has_frame_np.sum() == 0:
        return mx.zeros(pae_prob.shape[:-3], dtype=pae_prob.dtype)

    N_d = has_frame_np.shape[-1]
    ptm_norm = calculate_normalization(N_d)
    bin_center = get_bin_centers(min_bin, max_bin, no_bins).astype(pae_prob.dtype)
    per_bin_weight = 1.0 / (1.0 + (bin_center / ptm_norm) ** 2)  # [B]

    token_token_ptm = mx.sum(pae_prob * per_bin_weight, axis=-1)  # [..., N_d, N_d]
    per_token = mx.mean(token_token_ptm, axis=-1)  # [..., N_d]
    return _masked_max_over_frames(per_token, has_frame_np)


def calculate_iptm(
    pae_prob, has_frame, asym_id, min_bin, max_bin, no_bins, token_mask=None, eps=1e-8
):
    """ipTM. Same as ptm but only cross-chain token pairs contribute."""
    has_frame_np = np.asarray(has_frame, dtype=bool)
    asym_np = np.asarray(asym_id).astype(np.int64)
    if token_mask is not None:
        tm = np.asarray(token_mask, dtype=bool)
        idx = np.nonzero(tm)[0].astype(np.int32)
        pae_prob = mx.take(mx.take(pae_prob, mx.array(idx), axis=-3), mx.array(idx), axis=-2)
        has_frame_np = has_frame_np[idx]
        asym_np = asym_np[idx]

    if has_frame_np.sum() == 0:
        return mx.zeros(pae_prob.shape[:-3], dtype=pae_prob.dtype)

    N_d = has_frame_np.shape[-1]
    ptm_norm = calculate_normalization(N_d)
    bin_center = get_bin_centers(min_bin, max_bin, no_bins).astype(pae_prob.dtype)
    per_bin_weight = 1.0 / (1.0 + (bin_center / ptm_norm) ** 2)

    token_token_ptm = mx.sum(pae_prob * per_bin_weight, axis=-1)  # [..., N_d, N_d]

    is_diff_chain = (asym_np[None, :] != asym_np[:, None]).astype(np.float32)  # [N_d,N_d]
    is_diff_chain_mx = mx.array(is_diff_chain)
    denom = mx.array((eps + is_diff_chain.sum(axis=-1)).astype(np.float32))  # [N_d]
    per_token = mx.sum(token_token_ptm * is_diff_chain_mx, axis=-1) / denom  # [..., N_d]
    return _masked_max_over_frames(per_token, has_frame_np)


# ------------------------------------------------------------------------- af3 clash
def _remap_contiguous(asym_id: np.ndarray) -> np.ndarray:
    uniq = np.unique(asym_id)
    if len(uniq) != asym_id.max() + 1:
        remap = {int(o): n for n, o in enumerate(uniq)}
        return np.array([remap[int(x)] for x in asym_id], dtype=np.int64)
    return asym_id.astype(np.int64)


def calculate_clash(
    pred_coordinate,
    asym_id,
    atom_to_token_idx,
    is_polymer,
    threshold: float = 1.1,
):
    """AF3 complex clash flag per sample. Numpy host-side.

    Args:
        pred_coordinate: [N_sample, N_atom, 3]
        asym_id: [N_token]
        atom_to_token_idx: [N_atom]
        is_polymer: [N_atom]  (bool/0-1)
        threshold: af3 clash distance
    Returns:
        np.ndarray [N_sample] float32, 1.0 if any polymer-polymer chain pair clashes.
    """
    coords = np.asarray(pred_coordinate, dtype=np.float64)
    asym = _remap_contiguous(np.asarray(asym_id).astype(np.int64))
    a2t = np.asarray(atom_to_token_idx).astype(np.int64)
    is_poly = np.asarray(is_polymer).astype(bool)

    N_sample = coords.shape[0]
    N_chains = int(asym.max()) + 1

    # atom-level chain masks and chain type (lig if any atom is ligand else prot)
    chain_atom_mask = [(asym == c)[a2t] for c in range(N_chains)]
    is_ligand_atom = ~is_poly
    chain_is_lig = []
    for c in range(N_chains):
        m = chain_atom_mask[c]
        # torch asserts a single atom_type per chain; ligand==not polymer
        chain_is_lig.append(bool(is_ligand_atom[m].any()) if m.any() else False)

    out = np.zeros(N_sample, dtype=np.float32)
    for s in range(N_sample):
        clashed = False
        for i in range(N_chains):
            if chain_is_lig[i]:
                continue
            mi = chain_atom_mask[i]
            ni = int(mi.sum())
            ci = coords[s][mi]
            for j in range(i + 1, N_chains):
                if chain_is_lig[j]:
                    continue
                mj = chain_atom_mask[j]
                nj = int(mj.sum())
                cj = coords[s][mj]
                if ni == 0 or nj == 0:
                    continue
                d = np.linalg.norm(ci[:, None, :] - cj[None, :, :], axis=-1)
                total_clash = int((d < threshold).sum())
                relative_clash = total_clash / min(ni, nj)
                if total_clash > 100 or relative_clash > 0.5:
                    clashed = True
        out[s] = 1.0 if clashed else 0.0
    return out


# ------------------------------------------------------------------------- ranking
def ranking_score(
    pae_logits,
    has_frame,
    asym_id,
    atom_coordinate,
    atom_to_token_idx,
    atom_is_polymer,
    min_bin: float = 0.0,
    max_bin: float = 32.0,
    no_bins: int = 64,
    af3_clash_threshold: float = 1.1,
):
    """Per-sample ranking_score = 0.8*iptm + 0.2*ptm + 0.5*disorder - 100*has_clash.

    disorder is always zeros (mirrors torch). Returns np.ndarray [N_sample].
    Also returns the component dict for inspection.
    """
    _, pae_prob = logits_to_score(pae_logits, min_bin, max_bin, no_bins, return_prob=True)
    ptm = calculate_ptm(pae_prob, has_frame, min_bin, max_bin, no_bins)
    iptm = calculate_iptm(pae_prob, has_frame, asym_id, min_bin, max_bin, no_bins)
    ptm_np = np.asarray(ptm)
    iptm_np = np.asarray(iptm)
    disorder = np.zeros_like(ptm_np)
    has_clash = calculate_clash(
        atom_coordinate, asym_id, atom_to_token_idx, atom_is_polymer, af3_clash_threshold
    )
    score = 0.8 * iptm_np + 0.2 * ptm_np + 0.5 * disorder - 100.0 * has_clash
    return score, {
        "ptm": ptm_np,
        "iptm": iptm_np,
        "disorder": disorder,
        "has_clash": has_clash,
        "ranking_score": score,
    }
