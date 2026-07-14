"""torch<->MLX parity for the diffusion-transformer primitives on real weights:

  - AdaptiveLayerNorm            (opendde.model.modules.primitives)
  - ConditionedTransitionBlock   (opendde.model.modules.transformer)
  - AttentionPairBias (has_s=True AdaLN path, non-local token attention)

Weights are pulled from the diffusion_transformer block-0 subtree of the real
checkpoint. Random fp32 s / a / z fed identically into both paths.

Run:
  source .../conda.sh && conda activate of3mlx && cd .../OpenDDE \
  && export LAYERNORM_TYPE=torch OPENDDE_FOLDCP_MODE=single \
  && PYTHONPATH=. python tests/parity_diff_primitives.py
"""
import os
os.environ.setdefault("LAYERNORM_TYPE", "torch")
os.environ.setdefault("OPENDDE_FOLDCP_MODE", "single")

import numpy as np
import torch
import mlx.core as mx

from opendde.model.modules.primitives import AdaptiveLayerNorm
from opendde.model.modules.transformer import (
    AttentionPairBias,
    ConditionedTransitionBlock,
)
from opendde_mlx.diff_primitives import (
    adaptive_layer_norm,
    attention_pair_bias_adaln,
    conditioned_transition_block,
)
from opendde_mlx.weights import load_state_dict, subtree

CKPT = "/Users/tgg/.cache/opendde/checkpoint/opendde.pt"

C_A, C_S, C_Z, N_HEADS = 768, 384, 128, 16
N_TOKEN = 12


def rel(a_t, b_m):
    a = a_t.detach().to(torch.float32).numpy()
    b = np.array(b_m, dtype=np.float32)
    d = np.abs(a - b).max()
    r = d / (np.abs(a).max() + 1e-9)
    return float(d), float(r)


def to_t(sd_sub):
    return {k: torch.tensor(np.array(v)) for k, v in sd_sub.items()}


def main():
    mx.set_default_device(mx.gpu)
    sd = load_state_dict(CKPT)

    blk = "diffusion_module.diffusion_transformer.blocks.0"
    apb_sd = subtree(sd, f"{blk}.attention_pair_bias")
    ctb_sd = subtree(sd, f"{blk}.conditioned_transition_block")
    adaln_sd = subtree(ctb_sd, "adaln")

    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((N_TOKEN, C_A)).astype(np.float32)
    s_np = rng.standard_normal((N_TOKEN, C_S)).astype(np.float32)
    z_np = rng.standard_normal((N_TOKEN, N_TOKEN, C_Z)).astype(np.float32)

    a_t, s_t, z_t = torch.tensor(a_np), torch.tensor(s_np), torch.tensor(z_np)
    a_m, s_m, z_m = mx.array(a_np), mx.array(s_np), mx.array(z_np)

    print(f"{'step':34s} {'max_abs_diff':>13s} {'rel':>11s}")
    results = {}

    # ---- AdaptiveLayerNorm --------------------------------------------------
    aln = AdaptiveLayerNorm(c_a=C_A, c_s=C_S).eval()
    miss, unexp = aln.load_state_dict(to_t(adaln_sd), strict=False)
    print(f"# adaln load: missing={list(miss)} unexpected={list(unexp)}")
    with torch.no_grad():
        o = aln(a_t, s_t)
    m = adaptive_layer_norm(a_m, s_m, adaln_sd)
    mx.eval(m)
    results["AdaptiveLayerNorm"] = rel(o, m)

    # ---- ConditionedTransitionBlock ----------------------------------------
    ctb = ConditionedTransitionBlock(c_a=C_A, c_s=C_S, n=2, biasinit=-2.0).eval()
    miss, unexp = ctb.load_state_dict(to_t(ctb_sd), strict=False)
    print(f"# ctb load:   missing={list(miss)} unexpected={list(unexp)}")
    with torch.no_grad():
        o = ctb(a_t, s_t)
    m = conditioned_transition_block(a_m, s_m, ctb_sd)
    mx.eval(m)
    results["ConditionedTransitionBlock"] = rel(o, m)

    # ---- AttentionPairBias (has_s=True AdaLN, non-local) --------------------
    apb = AttentionPairBias(
        has_s=True, n_heads=N_HEADS, c_a=C_A, c_s=C_S, c_z=C_Z,
        biasinit=-2.0, cross_attention_mode=False,
    ).eval()
    miss, unexp = apb.load_state_dict(to_t(apb_sd), strict=False)
    print(f"# apb load:   missing={list(miss)} unexpected={list(unexp)}")

    # Sub-step parity: AdaLN inside APB, and the pair bias tensor.
    with torch.no_grad():
        a_ln_t = apb.layernorm_a(a=a_t, s=s_t)
    a_ln_m = adaptive_layer_norm(a_m, s_m, subtree(apb_sd, "layernorm_a"))
    mx.eval(a_ln_m)
    results["  APB.adaln"] = rel(a_ln_t, a_ln_m)

    from opendde.model.utils import permute_final_dims
    with torch.no_grad():
        bias_t = permute_final_dims(apb.linear_nobias_z(apb.layernorm_z(z_t)), [2, 0, 1])
    from opendde_mlx.modules import layer_norm as _ln, linear as _lin
    zb = _ln(z_m, apb_sd["layernorm_z.weight"], None, 1e-5)
    zb = _lin(zb, apb_sd["linear_nobias_z.weight"])
    zb = mx.moveaxis(zb, -1, -3)
    mx.eval(zb)
    results["  APB.pair_bias"] = rel(bias_t, zb)

    with torch.no_grad():
        o = apb(a_t, s_t, z_t)  # n_queries=None -> standard (non-local) path
    m = attention_pair_bias_adaln(a_m, s_m, z_m, apb_sd, no_heads=N_HEADS)
    mx.eval(m)
    results["AttentionPairBias"] = rel(o, m)

    print("-" * 60)
    worst = 0.0
    for name, (d, r) in results.items():
        print(f"{name:34s} {d:13.3e} {r:11.3e}")
        if not name.startswith("  "):
            worst = max(worst, r)
    print("-" * 60)
    print(f"{'WORST top-level rel':34s} {'':13s} {worst:11.3e}")
    print("PASS" if worst < 1e-3 else "FAIL")


if __name__ == "__main__":
    main()
