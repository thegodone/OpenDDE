"""Parity test: StructuralTokenExpander (full mode) + structural_token_refiner
MLX port vs torch, with real checkpoint weights.

Run:
  source .../conda.sh && conda activate of3mlx && cd .../OpenDDE
  export LAYERNORM_TYPE=torch OPENDDE_FOLDCP_MODE=single
  PYTHONPATH=. python opendde_mlx/tests/parity_structural.py
"""

import numpy as np
import torch
import mlx.core as mx

from opendde.model.modules.structural_tokens import StructuralTokenExpander
from opendde.model.modules.pairformer import PairformerStack

from opendde_mlx.weights import load_state_dict, subtree, remap_pairformer_stack
from opendde_mlx import structural as S

CKPT = "/Users/tgg/.cache/opendde/checkpoint/opendde.pt"

C_S = 384
C_Z = 384
C_S_INPUTS = 449
N_ROLES = 7


def rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.max(np.abs(a - b))
    den = np.max(np.abs(b)) + 1e-12
    return num, num / den


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    n_res = 12
    n_struct = 20

    # ---- random integer structural features -----------------------------
    parent = np.random.randint(0, n_res, size=(n_struct,)).astype(np.int64)
    role = np.random.randint(0, N_ROLES, size=(n_struct,)).astype(np.int64)
    residue_index = np.random.randint(0, 50, size=(n_res,)).astype(np.int64)
    asym_id = np.random.randint(0, 3, size=(n_res,)).astype(np.int64)
    prev_parent = np.random.randint(-1, n_res, size=(n_struct,)).astype(np.int64)
    next_parent = np.random.randint(-1, n_res, size=(n_struct,)).astype(np.int64)

    input_feature_dict = {
        "parent_residue_idx": torch.tensor(parent),
        "subtoken_role_id": torch.tensor(role),
        "residue_index": torch.tensor(residue_index),
        "asym_id": torch.tensor(asym_id),
        "prev_parent_residue_idx": torch.tensor(prev_parent),
        "next_parent_residue_idx": torch.tensor(next_parent),
    }

    s_inputs_res = np.random.randn(n_res, C_S_INPUTS).astype(np.float32)
    s_res = np.random.randn(n_res, C_S).astype(np.float32)
    z_res = np.random.randn(n_res, n_res, C_Z).astype(np.float32)

    # ---- torch expander (real weights) ----------------------------------
    sd = load_state_dict(CKPT)
    exp_sd_mlx = subtree(sd, "structural_token_expander")

    exp = StructuralTokenExpander(
        c_s=C_S, c_z=C_Z, c_s_inputs=C_S_INPUTS, n_roles=N_ROLES,
        init_mode="scratch", pair_projection_mode="full", pair_chunk_size=128,
    ).eval()
    # convert mlx arrays -> torch state dict
    exp_sd_torch = {k: torch.tensor(np.asarray(v)) for k, v in exp_sd_mlx.items()}
    missing, unexpected = exp.load_state_dict(exp_sd_torch, strict=False)
    print("[expander] missing:", list(missing))
    print("[expander] unexpected:", list(unexpected))

    with torch.no_grad():
        t_si, t_s, t_z, t_pf = exp(
            input_feature_dict=input_feature_dict,
            s_inputs_res=torch.tensor(s_inputs_res),
            s_res=torch.tensor(s_res),
            z_res=torch.tensor(z_res),
        )
    t_attn = t_pf["structural_pair_attn_bias"]

    # ---- mlx expander ---------------------------------------------------
    feats = {
        "parent_residue_idx": mx.array(parent),
        "subtoken_role_id": mx.array(role),
        "asym_id": mx.array(asym_id),
        "prev_parent_residue_idx": mx.array(prev_parent),
        "next_parent_residue_idx": mx.array(next_parent),
    }
    m_si, m_s, m_z, m_pf = S.structural_token_expander(
        exp_sd_mlx, feats,
        mx.array(s_inputs_res), mx.array(s_res), mx.array(z_res),
    )
    m_attn = m_pf["structural_pair_attn_bias"]

    print("\n=== EXPANDER sub-steps ===")
    for name, t, m in [
        ("s_inputs_struct", t_si, m_si),
        ("s_struct", t_s, m_s),
        ("z_struct", t_z, m_z),
        ("attn_bias", t_attn, m_attn),
    ]:
        num, r = rel(np.asarray(m), t.numpy())
        print(f"  {name:16s} max_abs={num:.3e} rel={r:.3e}")

    # ---- torch refiner (real weights) -----------------------------------
    ref_sd_mlx = subtree(sd, "structural_token_refiner")
    ref = PairformerStack(
        n_blocks=4, n_heads=8, c_z=C_Z, c_s=C_S,
        num_intermediate_factor=2, hidden_scale_up=True,
    ).eval()
    ref_sd_torch = {k: torch.tensor(np.asarray(v)) for k, v in ref_sd_mlx.items()}
    r_missing, r_unexpected = ref.load_state_dict(ref_sd_torch, strict=False)
    print("\n[refiner] missing:", list(r_missing))
    print("[refiner] unexpected:", list(r_unexpected))

    with torch.no_grad():
        t_s_out, t_z_out = ref(
            s=t_s, z=t_z, pair_mask=None,
            extra_attn_bias=t_attn,
        )

    # ---- mlx refiner ----------------------------------------------------
    ref_p, n_blocks = remap_pairformer_stack(sd, "structural_token_refiner")
    m_s_out, m_z_out = S.structural_token_refiner(
        m_s, m_z, ref_p, n_blocks=n_blocks, extra_attn_bias=m_attn,
    )

    print("\n=== REFINER outputs ===")
    for name, t, m in [("s_out", t_s_out, m_s_out), ("z_out", t_z_out, m_z_out)]:
        num, r = rel(np.asarray(m), t.numpy())
        print(f"  {name:8s} max_abs={num:.3e} rel={r:.3e}")

    # final gate
    _, rs = rel(np.asarray(m_s_out), t_s_out.numpy())
    _, rz = rel(np.asarray(m_z_out), t_z_out.numpy())
    _, ra = rel(np.asarray(m_attn), t_attn.numpy())
    _, rzs = rel(np.asarray(m_z), t_z.numpy())
    worst = max(rs, rz, ra, rzs)
    print(f"\nWORST rel (expander z/attn + refiner s/z) = {worst:.3e}")
    print("PASS" if worst < 1e-3 else "FAIL")


if __name__ == "__main__":
    main()
