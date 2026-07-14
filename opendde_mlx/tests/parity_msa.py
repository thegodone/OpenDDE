"""torch<->MLX parity for the OpenDDE MSA module on real checkpoint weights.

Gates:
  (1) OuterProductMean            (opendde/model/triangular/layers.py)
  (2) MSAPairWeightedAveraging    (sub-step of MSABlock)
  (3) MSAStack                    (mpwa + transition_m)
  (4) MSABlock (block 0)          full Boltz-order block
  (5) msa_module_blocks (4 blk)   full stack, returns updated z

Uses a dummy 1-row MSA (N_seq=1) so no MSA sampling / one-hot logic is needed.

Run:
  source .../conda.sh && conda activate of3mlx && cd .../OpenDDE \
  && export LAYERNORM_TYPE=torch OPENDDE_FOLDCP_MODE=single \
  && PYTHONPATH=. python tests/parity_msa.py
"""
import os
os.environ.setdefault("LAYERNORM_TYPE", "torch")
os.environ.setdefault("OPENDDE_FOLDCP_MODE", "single")

import numpy as np
import torch
import mlx.core as mx

from opendde.model.triangular.layers import OuterProductMean
from opendde.model.modules.pairformer import (
    MSAPairWeightedAveraging,
    MSAStack,
    MSABlock,
    MSAModule,
)
from opendde_mlx.msa import (
    outer_product_mean,
    msa_pair_weighted_averaging,
    msa_stack,
    msa_block,
    msa_module_blocks,
)
from opendde_mlx.weights import load_state_dict, subtree

CKPT = "/Users/tgg/.cache/opendde/checkpoint/opendde.pt"

C_M = 128
C_Z = 384
N_SEQ = 1     # dummy 1-row MSA
N_RES = 7
N_HEADS_PAIR = 12  # hidden_scale_up: c_z // 32


def rel(a_t, b_m):
    a = a_t.detach().float().numpy()
    b = np.array(b_m)
    d = np.abs(a - b).max()
    r = d / (np.abs(a).max() + 1e-9)
    return float(d), float(r)


def to_t(x):
    return torch.tensor(np.array(x))


def sub_torch(sd, prefix):
    """torch-tensor state dict for a subtree (prefix stripped)."""
    if not prefix.endswith("."):
        prefix += "."
    return {k[len(prefix):]: to_t(v) for k, v in sd.items() if k.startswith(prefix)}


def main():
    mx.set_default_device(mx.gpu)
    torch.manual_seed(0)
    sd = load_state_dict(CKPT)
    b0 = subtree(sd, "msa_module.blocks.0")

    print(f"{'step':34s} {'max_abs_diff':>13s} {'rel':>12s}")
    results = {}

    rng = np.random.default_rng(11)
    m_np = rng.standard_normal((N_SEQ, N_RES, C_M)).astype(np.float32)
    z_np = rng.standard_normal((N_RES, N_RES, C_Z)).astype(np.float32)
    pair_mask_np = np.ones((N_RES, N_RES), dtype=np.float32)

    # ---------------------------------------------------------------- (1) OPM
    opm = OuterProductMean(c_m=C_M, c_z=C_Z, c_hidden=32).eval()
    miss, unexp = opm.load_state_dict(sub_torch(sd, "msa_module.blocks.0.outer_product_mean_msa"), strict=False)
    print(f"# OPM load: missing={list(miss)} unexpected={list(unexp)}")
    with torch.no_grad():
        o_t = opm(to_t(m_np))
    o_m = outer_product_mean(mx.array(m_np), subtree(b0, "outer_product_mean_msa"))
    mx.eval(o_m)
    results["outer_product_mean"] = rel(o_t, o_m)

    # ------------------------------------------------ (2) MSAPairWeightedAveraging
    mpwa = MSAPairWeightedAveraging(c_m=C_M, c=8, c_z=C_Z, n_heads=8).eval()
    miss, unexp = mpwa.load_state_dict(
        sub_torch(sd, "msa_module.blocks.0.msa_stack.msa_pair_weighted_averaging"), strict=False
    )
    print(f"# MPWA load: missing={list(miss)} unexpected={list(unexp)}")
    with torch.no_grad():
        p_t = mpwa(to_t(m_np), to_t(z_np))
    p_m = msa_pair_weighted_averaging(
        mx.array(m_np), mx.array(z_np),
        subtree(b0, "msa_stack.msa_pair_weighted_averaging"),
    )
    mx.eval(p_m)
    results["msa_pair_weighted_averaging"] = rel(p_t, p_m)

    # ----------------------------------------------------------- (3) MSAStack
    stack = MSAStack(c_m=C_M, c_z=C_Z).eval()
    miss, unexp = stack.load_state_dict(sub_torch(sd, "msa_module.blocks.0.msa_stack"), strict=False)
    print(f"# MSAStack load: missing={list(miss)} unexpected={list(unexp)}")
    with torch.no_grad():
        # inference_forward mutates m in place -> clone
        s_t = stack(to_t(m_np).clone(), to_t(z_np))
    s_m = msa_stack(mx.array(m_np), mx.array(z_np), subtree(b0, "msa_stack"))
    mx.eval(s_m)
    results["msa_stack"] = rel(s_t, s_m)

    # --------------------------------------------------------- (4) MSABlock 0
    block = MSABlock(c_m=C_M, c_z=C_Z, is_last_block=False, hidden_scale_up=True).eval()
    miss, unexp = block.load_state_dict(sub_torch(sd, "msa_module.blocks.0"), strict=False)
    print(f"# MSABlock load: missing={list(miss)} unexpected={list(unexp)}")
    with torch.no_grad():
        mb_t, zb_t = block(
            to_t(m_np).clone(), to_t(z_np).clone(), to_t(pair_mask_np),
            triangle_multiplicative="torch", triangle_attention="torch",
        )
    mb_m, zb_m = msa_block(
        mx.array(m_np), mx.array(z_np), b0, mx.array(pair_mask_np),
        is_last=False, no_heads_pair=N_HEADS_PAIR,
    )
    mx.eval(mb_m, zb_m)
    results["msa_block.m"] = rel(mb_t, mb_m)
    results["msa_block.z"] = rel(zb_t, zb_m)

    # ------------------------------------------------- (5) full 4-block stack
    # Build a real MSAModule (needs msa_configs.msa_depth) and run its 4 blocks
    # directly on a pre-projected msa_sample (skip feature/one-hot prep).
    mod = MSAModule(
        n_blocks=4, c_m=C_M, c_z=C_Z, c_s_inputs=449,
        hidden_scale_up=True, msa_configs={"msa_depth": 8},
    ).eval()
    miss, unexp = mod.load_state_dict(subtree_to_torch(sd, "msa_module"), strict=False)
    print(f"# MSAModule load: missing={len(miss)} unexpected={len(unexp)}")

    m_full = mx.array(m_np)
    z_full = mx.array(z_np)
    pm = mx.array(pair_mask_np)
    # torch: replicate MSAModule.forward block loop (checkpoint_blocks, no grad)
    mt = to_t(m_np).clone()
    zt = to_t(z_np).clone()
    pmt = to_t(pair_mask_np)
    with torch.no_grad():
        for blk in mod.blocks:
            mt, zt = blk(mt, zt, pmt,
                         triangle_multiplicative="torch", triangle_attention="torch")
    z_stack_m = msa_module_blocks(
        m_full, z_full, subtree(sd, "msa_module"), pm,
        n_blocks=4, no_heads_pair=N_HEADS_PAIR,
    )
    mx.eval(z_stack_m)
    results["msa_module_blocks.z"] = rel(zt, z_stack_m)

    print("-" * 60)
    ok = True
    for k, (d, r) in results.items():
        flag = "OK" if r < 1e-3 else "FAIL"
        if r >= 1e-3:
            ok = False
        print(f"{k:34s} {d:13.3e} {r:12.3e}  {flag}")
    print("-" * 60)
    print("ALL PASS" if ok else "SOME FAILED")


def subtree_to_torch(sd, prefix):
    if not prefix.endswith("."):
        prefix += "."
    return {k[len(prefix):]: to_t(v) for k, v in sd.items() if k.startswith(prefix)}


if __name__ == "__main__":
    main()
