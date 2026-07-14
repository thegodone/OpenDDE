"""Step-by-step torch<->MLX parity for the OpenDDE Pairformer block, real weights.

Instantiates the torch PairformerBlock (c_z=384, c_s=384, hidden_scale_up=True),
loads real block-0 weights, and compares EACH sub-step against the reused MLX
primitives on identical fp32 input. Also the full block and 48-block stack.
"""
import os
os.environ.setdefault("LAYERNORM_TYPE", "torch")

import numpy as np
import torch
import mlx.core as mx

from opendde.model.modules.pairformer import PairformerBlock, PairformerStack
from opendde_mlx import modules as M
from opendde_mlx.weights import (load_state_dict, subtree, remap_pairformer_block,
                                 remap_pairformer_stack)

CKPT = "/Users/tgg/.cache/opendde/checkpoint/opendde.pt"
C = 384
HP, HPB = 12, 16   # tri-att heads, pair-bias heads


def t2m(t): return mx.array(t.detach().to(torch.float32).numpy())
def rel(a_t, b_m):
    a = a_t.detach().numpy(); b = np.array(b_m)
    d = np.abs(a - b).max(); r = d / (np.abs(a).max() + 1e-9)
    return d, r


def raw_torch_sd():
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = obj["model"]
    return {k[len("module."):]: v for k, v in sd.items() if k.startswith("module.")}


def main():
    mx.set_default_device(mx.gpu)
    torch.manual_seed(0)
    raw = raw_torch_sd()
    mlx_sd = load_state_dict(CKPT)

    # torch reference block 0
    blk = PairformerBlock(n_heads=HPB, c_z=C, c_s=C, hidden_scale_up=True).eval()
    b0_torch = {k[len("pairformer_stack.blocks.0."):]: v
                for k, v in raw.items() if k.startswith("pairformer_stack.blocks.0.")}
    missing, unexpected = blk.load_state_dict(b0_torch, strict=False)
    missing = [m for m in missing if "foldcp" not in m.lower()]
    print(f"torch block load: missing={missing[:3]} unexpected={unexpected[:3]}")

    # mlx params (remapped) for block 0
    p = remap_pairformer_block(subtree(mlx_sd, "pairformer_stack.blocks.0"))

    # inputs
    N = 40
    rng = np.random.default_rng(1)
    z_np = (rng.standard_normal((N, N, C)) * 0.3).astype(np.float32)
    s_np = (rng.standard_normal((N, C)) * 0.3).astype(np.float32)
    pm_np = np.ones((N, N), np.float32)
    zt = torch.tensor(z_np); st = torch.tensor(s_np); pmt = torch.tensor(pm_np)
    zm = mx.array(z_np); sm = mx.array(s_np); pmm = mx.array(pm_np)

    print(f"\n{'step':28s} {'max_abs_diff':>13s} {'rel':>11s}")
    results = {}
    with torch.no_grad():
        # 1 tri_mul_out
        o = blk.tri_mul_out(zt, pmt, triangle_multiplicative="torch")
        m = M.tri_mul(zm, M._sub(p, "pair_stack.tri_mul_out"), True, pmm)
        results["tri_mul_out"] = rel(o, m)
        # 2 tri_mul_in
        o = blk.tri_mul_in(zt, pmt, triangle_multiplicative="torch")
        m = M.tri_mul(zm, M._sub(p, "pair_stack.tri_mul_in"), False, pmm)
        results["tri_mul_in"] = rel(o, m)
        # 3 tri_att_start
        o = blk.tri_att_start(zt, mask=pmt)
        m = M.triangle_attention(zm, M._sub(p, "pair_stack.tri_att_start"), HP, True, pmm, inf=1e9)
        results["tri_att_start"] = rel(o, m)
        # 4 tri_att_end (no internal transpose; feed same z)
        o = blk.tri_att_end(zt, mask=pmt)
        m = M.triangle_attention(zm, M._sub(p, "pair_stack.tri_att_end"), HP, True, pmm, inf=1e9)
        results["tri_att_end"] = rel(o, m)
        # 5 pair_transition (SwiGLU, no mask)
        o = blk.pair_transition(zt)
        m = M.swiglu_transition(zm, M._sub(p, "pair_stack.pair_transition"), mask=None)
        results["pair_transition"] = rel(o, m)
        # 6 attention_pair_bias (has_s=False: a=s, s=None)
        o = blk.attention_pair_bias(a=st, s=None, z=zt)
        m = M.attention_pair_bias(sm, zm, M._sub(p, "attn_pair_bias"), HPB, mask=None)
        results["attention_pair_bias"] = rel(o, m)
        # 7 single_transition
        o = blk.single_transition(st)
        m = M.swiglu_transition(sm, M._sub(p, "single_transition"), mask=None)
        results["single_transition"] = rel(o, m)
        # 8 full block
        so, zo = blk(st, zt, pair_mask=pmt)
        sm2, zm2 = M.pairformer_block(sm, zm, p, pmm, pmm,
                                      no_heads_pair_bias=HPB, no_heads_pair=HP, inf=1e9)
        results["block.s"] = rel(so, sm2)
        results["block.z"] = rel(zo, zm2)

    for k, (d, r) in results.items():
        print(f"{k:28s} {d:13.3e} {r:11.3e}")

    # 48-block stack
    stack = PairformerStack(n_blocks=48, n_heads=HPB, c_z=C, c_s=C, hidden_scale_up=True).eval()
    sd_stack = {k[len("pairformer_stack."):]: v for k, v in raw.items()
                if k.startswith("pairformer_stack.")}
    stack.load_state_dict(sd_stack, strict=False)
    with torch.no_grad():
        so, zo = stack(st.clone(), zt.clone(), pair_mask=pmt)
    p_stack, nb = remap_pairformer_stack(mlx_sd, "pairformer_stack")
    sm2, zm2 = M.pairformer_stack(sm, zm, p_stack, pmm, pmm, n_blocks=nb,
                                  no_heads_pair_bias=HPB, no_heads_pair=HP, inf=1e9)
    mx.eval(sm2, zm2)
    ds, rs = rel(so, sm2); dz, rz = rel(zo, zm2)
    print(f"{'STACK(48).s':28s} {ds:13.3e} {rs:11.3e}")
    print(f"{'STACK(48).z':28s} {dz:13.3e} {rz:11.3e}")

    worst = max(r for _, r in results.values())
    worst = max(worst, rs, rz)
    print(f"\nSTEP-PARITY {'PASS' if worst < 1e-3 else 'FAIL'} (worst rel {worst:.2e})")
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
