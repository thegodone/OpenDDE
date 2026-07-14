"""Torch-vs-MLX parity for DistogramHead and ConfidenceHead (single-device path).

Loads the REAL opendde.pt weights for the ``distogram_head`` and
``confidence_head`` subtrees, builds identical random fp32 inputs, runs torch
(no_grad, CPU) and the MLX port, and prints per-sub-step + final max_abs_diff
and rel errors.  Target: rel < 1e-3.

Run:
  source .../conda.sh && conda activate of3mlx && cd .../OpenDDE
  export LAYERNORM_TYPE=torch OPENDDE_FOLDCP_MODE=single
  PYTHONPATH=. python tests/parity_heads.py
"""

import sys

import numpy as np
import mlx.core as mx
import torch

from opendde_mlx.weights import load_state_dict, subtree, remap_pairformer_stack
from opendde_mlx import heads as H

CKPT = "/Users/tgg/.cache/opendde/checkpoint/opendde.pt"

C_S = 384
C_Z = 384
C_S_INPUTS = 449
NO_BINS = 96
MAX_ATOMS_PER_TOKEN = 24

N_TOKEN = 8
N_ATOM = 20
N_SAMPLE = 2


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.max(np.abs(a - b)))
    den = float(np.max(np.abs(b))) + 1e-12
    return num / den


def main() -> int:
    torch.manual_seed(0)
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(0)

    print("loading checkpoint ...")
    sd_mx = load_state_dict(CKPT)

    # ---- import torch heads ----
    from opendde.model.modules.head import DistogramHead
    from opendde.model.modules.confidence import ConfidenceHead

    # =====================================================================
    # DistogramHead
    # =====================================================================
    dist_head = DistogramHead(c_z=C_Z, no_bins=NO_BINS).eval()
    raw0 = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    sd_dist_t = {}
    for k, v in raw0.items():
        k2 = k.replace("module.", "")
        if k2.startswith("distogram_head."):
            sd_dist_t[k2[len("distogram_head."):]] = v.float()
    miss = dist_head.load_state_dict(sd_dist_t, strict=False)
    print(f"[distogram] missing={list(miss.missing_keys)} unexpected={list(miss.unexpected_keys)}")

    z_np = rng.standard_normal((N_TOKEN, N_TOKEN, C_Z)).astype(np.float32) * 0.5
    with torch.no_grad():
        d_t = dist_head(torch.from_numpy(z_np)).numpy()
    p_dist = subtree(sd_mx, "distogram_head")
    d_m = np.array(H.distogram_head(mx.array(z_np), p_dist))
    r_dist = rel_err(d_m, d_t)
    print(f"[distogram] logits shape {d_t.shape} rel={r_dist:.3e} maxabs={np.max(np.abs(d_m-d_t)):.3e}")

    # =====================================================================
    # ConfidenceHead
    # =====================================================================
    conf = ConfidenceHead(
        n_blocks=4, c_s=C_S, c_z=C_Z, c_s_inputs=C_S_INPUTS,
        b_pae=64, b_pde=64, b_plddt=50, b_resolved=2,
        max_atoms_per_token=MAX_ATOMS_PER_TOKEN, blocks_per_ckpt=None,
        distance_bin_start=3.25, distance_bin_end=52.0, distance_bin_step=1.25,
        hidden_scale_up=True,
    ).eval()
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    sd_conf_t = {}
    for k, v in raw.items():
        k2 = k.replace("module.", "")
        if k2.startswith("confidence_head."):
            sd_conf_t[k2[len("confidence_head."):]] = v.float()
    miss = conf.load_state_dict(sd_conf_t, strict=False)
    print(f"[confidence] missing={list(miss.missing_keys)} unexpected={list(miss.unexpected_keys)}")

    # ---- inputs ----
    s_inputs = rng.standard_normal((N_TOKEN, C_S_INPUTS)).astype(np.float32) * 0.5
    s_trunk = rng.standard_normal((N_TOKEN, C_S)).astype(np.float32) * 0.5
    z_trunk = rng.standard_normal((N_TOKEN, N_TOKEN, C_Z)).astype(np.float32) * 0.5
    pair_mask = np.ones((N_TOKEN, N_TOKEN), dtype=np.float32)
    x_pred = (rng.standard_normal((N_SAMPLE, N_ATOM, 3)).astype(np.float32) * 10.0)

    # rep-atom mask: exactly N_TOKEN atoms are representatives
    rep_mask = np.zeros((N_ATOM,), dtype=bool)
    rep_mask[rng.permutation(N_ATOM)[:N_TOKEN]] = True
    atom_to_token = rng.integers(0, N_TOKEN, size=(N_ATOM,)).astype(np.int64)
    atom_to_tokatom = rng.integers(0, MAX_ATOMS_PER_TOKEN, size=(N_ATOM,)).astype(np.int64)

    feats_t = {
        "distogram_rep_atom_mask": torch.from_numpy(rep_mask),
        "atom_to_token_idx": torch.from_numpy(atom_to_token),
        "atom_to_tokatom_idx": torch.from_numpy(atom_to_tokatom),
    }

    with torch.no_grad():
        plddt_t, pae_t, pde_t, resolved_t = conf(
            input_feature_dict=feats_t,
            s_inputs=torch.from_numpy(s_inputs),
            s_trunk=torch.from_numpy(s_trunk),
            z_trunk=torch.from_numpy(z_trunk),
            pair_mask=torch.from_numpy(pair_mask),
            x_pred_coords=torch.from_numpy(x_pred),
            triangle_multiplicative="torch",
            triangle_attention="torch",
        )
    plddt_t = plddt_t.numpy(); pae_t = pae_t.numpy()
    pde_t = pde_t.numpy(); resolved_t = resolved_t.numpy()

    # ---- MLX ----
    cp = subtree(sd_mx, "confidence_head")
    pf, n_blocks = remap_pairformer_stack(cp, "pairformer_stack")
    feats_m = {
        "distogram_rep_atom_mask": mx.array(rep_mask),
        "atom_to_token_idx": mx.array(atom_to_token),
        "atom_to_tokatom_idx": mx.array(atom_to_tokatom),
    }
    plddt_m, pae_m, pde_m, resolved_m = H.confidence_head(
        feats_m,
        mx.array(s_inputs), mx.array(s_trunk), mx.array(z_trunk),
        mx.array(pair_mask), mx.array(x_pred),
        cp, pf, n_blocks,
    )
    plddt_m = np.array(plddt_m); pae_m = np.array(pae_m)
    pde_m = np.array(pde_m); resolved_m = np.array(resolved_m)

    print(f"[confidence] n_blocks remapped = {n_blocks}")
    r_pae = rel_err(pae_m, pae_t)
    r_pde = rel_err(pde_m, pde_t)
    r_plddt = rel_err(plddt_m, plddt_t)
    r_resolved = rel_err(resolved_m, resolved_t)
    print(f"[confidence] pae      shape {pae_t.shape} rel={r_pae:.3e} maxabs={np.max(np.abs(pae_m-pae_t)):.3e}")
    print(f"[confidence] pde      shape {pde_t.shape} rel={r_pde:.3e} maxabs={np.max(np.abs(pde_m-pde_t)):.3e}")
    print(f"[confidence] plddt    shape {plddt_t.shape} rel={r_plddt:.3e} maxabs={np.max(np.abs(plddt_m-plddt_t)):.3e}")
    print(f"[confidence] resolved shape {resolved_t.shape} rel={r_resolved:.3e} maxabs={np.max(np.abs(resolved_m-resolved_t)):.3e}")

    worst = max(r_dist, r_pae, r_pde, r_plddt, r_resolved)
    print(f"\nWORST rel = {worst:.3e}  ->  {'PASS' if worst < 1e-3 else 'FAIL'} (target 1e-3)")
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
