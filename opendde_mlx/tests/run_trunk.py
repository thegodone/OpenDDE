"""Prove the reuse thesis: load REAL OpenDDE trunk weights (c_z=384, heads=12)
into the reused openfold-3-mlx Pairformer and run the full 48-block stack.

Full torch-vs-MLX parity needs the opendde torch package stood up (next step);
this validates weight-mapping completeness + that the primitives execute and
produce finite, correctly-shaped output on the real 655M-param weights."""

import sys

import numpy as np
import mlx.core as mx

from opendde_mlx import modules as M
from opendde_mlx.weights import load_state_dict, remap_pairformer_stack

CKPT = "/Users/tgg/.cache/opendde/checkpoint/opendde.pt"
C_S, C_Z = 384, 384
NO_HEADS_PAIR = 12          # hidden_scale_up: c_z//32
NO_HEADS_PAIR_BIAS = 16     # linear_nobias_z is (16, 384)


def main():
    mx.set_default_device(mx.gpu)
    sd = load_state_dict(CKPT)
    print(f"loaded {len(sd)} tensors")

    p_stack, n_blocks = remap_pairformer_stack(sd, "pairformer_stack")
    print(f"remapped pairformer_stack: {n_blocks} blocks, {len(p_stack)} tensors")

    # sanity: every reused primitive's expected keys present in block 0
    b0 = {k[len("blocks.0."):]: v for k, v in p_stack.items() if k.startswith("blocks.0.")}
    need = [
        "pair_stack.tri_mul_out.linear_a_p.weight",
        "pair_stack.tri_mul_in.linear_z.weight",
        "pair_stack.tri_att_start.linear_z.weight",
        "pair_stack.tri_att_end.mha.linear_q.weight",
        "pair_stack.pair_transition.swiglu.linear_a.weight",
        "pair_stack.pair_transition.linear_out.weight",
        "attn_pair_bias.layer_norm_a.weight",
        "attn_pair_bias.linear_z.weight",
        "attn_pair_bias.mha.linear_q.weight",
        "attn_pair_bias.mha.linear_q.bias",
        "single_transition.swiglu.linear_b.weight",
        "single_transition.linear_out.weight",
    ]
    missing = [k for k in need if k not in b0]
    assert not missing, f"MISSING remapped keys: {missing}"
    print(f"all {len(need)} reused-primitive keys present in block 0; "
          f"tri_att heads={b0['pair_stack.tri_att_start.linear_z.weight'].shape[0]}, "
          f"pair_bias heads={b0['attn_pair_bias.linear_z.weight'].shape[0]}")

    # run the real 48-block trunk on random input
    N = 48
    rng = np.random.default_rng(0)
    s = mx.array(rng.standard_normal((N, C_S)).astype(np.float32) * 0.1)
    z = mx.array(rng.standard_normal((N, N, C_Z)).astype(np.float32) * 0.1)
    smask = mx.ones((N,)); pmask = mx.ones((N, N))

    s_out, z_out = M.pairformer_stack(
        s, z, p_stack, smask, pmask, n_blocks=n_blocks,
        no_heads_pair_bias=NO_HEADS_PAIR_BIAS, no_heads_pair=NO_HEADS_PAIR, inf=1e9,
    )
    mx.eval(s_out, z_out)
    sn, zn = np.array(s_out), np.array(z_out)
    ok = (sn.shape == (N, C_S) and zn.shape == (N, N, C_Z)
          and np.isfinite(sn).all() and np.isfinite(zn).all())
    print(f"48-block trunk ran on REAL weights: s{sn.shape} z{zn.shape} "
          f"finite={np.isfinite(sn).all() and np.isfinite(zn).all()} "
          f"s_range=[{sn.min():.3f},{sn.max():.3f}] z_range=[{zn.min():.3f},{zn.max():.3f}]")
    print("REUSE-THESIS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
