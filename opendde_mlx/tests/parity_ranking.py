# SPDX-License-Identifier: Apache-2.0
# Parity: opendde_mlx.ranking  vs  opendde torch (sample_confidence + metrics.clash).
# Gates logits_to_score, calculate_ptm, calculate_iptm, af3 clash, and the full
# ranking_score scalar per sample + identical sample RANK ORDER.

import numpy as np
import torch
import mlx.core as mx

from opendde.model import sample_confidence as sc
from opendde_mlx import ranking as R

MIN_BIN, MAX_BIN, NO_BINS = 0.0, 32.0, 64
CLASH_THR = 1.1


def rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.abs(a - b).max()
    den = max(np.abs(b).max(), 1e-8)
    return float(num / den), float(num)


def build_inputs(seed=0):
    rng = np.random.default_rng(seed)
    N_sample = 5
    # 3 chains: chain0 (prot) tokens 0-3, chain1 (prot) 4-7, chain2 (lig) 8-11
    N_token = 12
    asym_id = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    has_frame = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)

    # 2 atoms per token -> 24 atoms
    atom_to_token_idx = np.repeat(np.arange(N_token), 2).astype(np.int64)
    N_atom = atom_to_token_idx.shape[0]
    # tokens 0-7 polymer, tokens 8-11 ligand
    token_is_poly = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    atom_is_polymer = token_is_poly[atom_to_token_idx]

    pae_logits = rng.standard_normal((N_sample, N_token, N_token, NO_BINS)).astype(np.float32)

    # Coordinates: chain0 near origin, chain1 far (+100 x), chain2 ligand (+200 x).
    coords = np.zeros((N_sample, N_atom, 3), dtype=np.float32)
    chain0_atoms = np.nonzero((asym_id == 0)[atom_to_token_idx])[0]
    chain1_atoms = np.nonzero((asym_id == 1)[atom_to_token_idx])[0]
    chain2_atoms = np.nonzero((asym_id == 2)[atom_to_token_idx])[0]
    for s in range(N_sample):
        base0 = rng.standard_normal((len(chain0_atoms), 3)).astype(np.float32) * 3.0
        coords[s, chain0_atoms] = base0
        coords[s, chain2_atoms] = rng.standard_normal((len(chain2_atoms), 3)).astype(np.float32) + 200.0
        if s in (1, 3):
            # clash: chain1 overlaps chain0
            coords[s, chain1_atoms] = base0 + 0.1
        else:
            coords[s, chain1_atoms] = base0 + np.array([100.0, 0, 0], dtype=np.float32)
    return dict(
        pae_logits=pae_logits,
        has_frame=has_frame,
        asym_id=asym_id,
        atom_to_token_idx=atom_to_token_idx,
        atom_is_polymer=atom_is_polymer,
        coords=coords,
    )


def main():
    inp = build_inputs()
    bp = dict(min_bin=MIN_BIN, max_bin=MAX_BIN, no_bins=NO_BINS)

    # ---- torch reference ------------------------------------------------------
    t_pae = torch.tensor(inp["pae_logits"])
    t_hf = torch.tensor(inp["has_frame"])
    t_asym = torch.tensor(inp["asym_id"])
    _, t_pae_prob = sc.logits_to_score(t_pae, return_prob=True, **bp)
    t_score_ls = sc.logits_to_score(t_pae, **bp)  # expected value [N_s,Nt,Nt]
    t_ptm = sc.calculate_ptm(t_pae_prob, has_frame=t_hf, **bp)
    t_iptm = sc.calculate_iptm(t_pae_prob, has_frame=t_hf, asym_id=t_asym, **bp)
    t_clash = sc.calculate_clash(
        torch.tensor(inp["coords"]),
        t_asym,
        torch.tensor(inp["atom_to_token_idx"]),
        torch.tensor(inp["atom_is_polymer"].astype(np.int64)),
        CLASH_THR,
    ).float()
    t_disorder = torch.zeros_like(t_ptm)
    t_rank = (0.8 * t_iptm + 0.2 * t_ptm + 0.5 * t_disorder - 100.0 * t_clash).numpy()

    # ---- mlx port -------------------------------------------------------------
    m_pae = mx.array(inp["pae_logits"])
    m_score_ls, m_pae_prob = R.logits_to_score(m_pae, return_prob=True, **bp)
    m_ptm = R.calculate_ptm(m_pae_prob, inp["has_frame"], **bp)
    m_iptm = R.calculate_iptm(m_pae_prob, inp["has_frame"], inp["asym_id"], **bp)
    m_rank, comps = R.ranking_score(
        m_pae,
        inp["has_frame"],
        inp["asym_id"],
        inp["coords"],
        inp["atom_to_token_idx"],
        inp["atom_is_polymer"],
        MIN_BIN, MAX_BIN, NO_BINS, CLASH_THR,
    )

    # ---- report ---------------------------------------------------------------
    r_ls = rel(np.asarray(m_score_ls), t_score_ls.numpy())
    r_ptm = rel(np.asarray(m_ptm), t_ptm.numpy())
    r_iptm = rel(np.asarray(m_iptm), t_iptm.numpy())
    r_clash = rel(comps["has_clash"], t_clash.numpy())
    r_rank = rel(m_rank, t_rank)

    print("=== per-sub-step rel (max_abs / rel) ===")
    print(f"logits_to_score : abs={r_ls[1]:.3e}  rel={r_ls[0]:.3e}")
    print(f"calculate_ptm   : abs={r_ptm[1]:.3e}  rel={r_ptm[0]:.3e}")
    print(f"calculate_iptm  : abs={r_iptm[1]:.3e}  rel={r_iptm[0]:.3e}")
    print(f"has_clash       : abs={r_clash[1]:.3e}  rel={r_clash[0]:.3e}")
    print(f"ranking_score   : abs={r_rank[1]:.3e}  rel={r_rank[0]:.3e}")
    print("torch ptm  :", np.round(t_ptm.numpy(), 5))
    print("mlx   ptm  :", np.round(np.asarray(m_ptm), 5))
    print("torch iptm :", np.round(t_iptm.numpy(), 5))
    print("mlx   iptm :", np.round(np.asarray(m_iptm), 5))
    print("torch clash:", t_clash.numpy())
    print("mlx   clash:", comps["has_clash"])
    print("torch rank :", np.round(t_rank, 5))
    print("mlx   rank :", np.round(m_rank, 5))

    t_order = np.argsort(-t_rank)
    m_order = np.argsort(-m_rank)
    print("torch rank order:", t_order)
    print("mlx   rank order:", m_order)

    final_rel = max(r_ls[0], r_ptm[0], r_iptm[0], r_clash[0], r_rank[0])
    order_ok = np.array_equal(t_order, m_order)
    print(f"\nMAX rel over all sub-steps = {final_rel:.3e}")
    print(f"rank order identical = {order_ok}")
    assert order_ok, "RANK ORDER MISMATCH"
    assert final_rel < 1e-3, f"rel {final_rel} >= 1e-3"
    print("PASS")


if __name__ == "__main__":
    main()
