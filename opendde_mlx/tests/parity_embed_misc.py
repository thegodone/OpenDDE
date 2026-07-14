"""torch<->MLX parity for FourierEmbedding + RelativePositionEncoding.generate_relp
(and the trailing linear_no_bias projection), on real checkpoint weights.

Run:
  source .../conda.sh && conda activate of3mlx && cd .../OpenDDE \
  && export LAYERNORM_TYPE=torch OPENDDE_FOLDCP_MODE=single \
  && PYTHONPATH=. python tests/parity_embed_misc.py
"""
import os
os.environ.setdefault("LAYERNORM_TYPE", "torch")

import numpy as np
import torch
import mlx.core as mx

from opendde.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from opendde_mlx.embed_misc import (
    fourier_embedding, generate_relp, relative_position_encoding,
)
from opendde_mlx.weights import load_state_dict, subtree

CKPT = "/Users/tgg/.cache/opendde/checkpoint/opendde.pt"


def rel(a_t, b_m):
    a = a_t.detach().numpy(); b = np.array(b_m)
    d = np.abs(a - b).max()
    r = d / (np.abs(a).max() + 1e-9)
    return d, r


def main():
    mx.set_default_device(mx.gpu)
    sd = load_state_dict(CKPT)

    print(f"{'step':28s} {'max_abs_diff':>13s} {'rel':>11s}")
    results = {}

    # ---- FourierEmbedding (c=256, real w/b buffers from checkpoint) ----------
    fe_prefix = "diffusion_module.diffusion_conditioning.fourier_embedding"
    fe_sd = subtree(sd, fe_prefix)          # {'w':.., 'b':..}
    c = fe_sd["w"].shape[0]
    fe = FourierEmbedding(c=c).eval()
    tsd = {"w": torch.tensor(np.array(fe_sd["w"])),
           "b": torch.tensor(np.array(fe_sd["b"]))}
    missing, unexpected = fe.load_state_dict(tsd, strict=False)
    print(f"# fourier load: missing={list(missing)} unexpected={list(unexpected)} c={c}")

    rng = np.random.default_rng(7)
    t_np = (rng.standard_normal((3, 5)) * 4.0).astype(np.float32)   # [..., N_sample]
    with torch.no_grad():
        o = fe(torch.tensor(t_np))
    m = fourier_embedding(mx.array(t_np), {"w": fe_sd["w"], "b": fe_sd["b"]})
    mx.eval(m)
    results["fourier_embedding"] = rel(o, m)

    # ---- RelativePositionEncoding: generate_relp + linear_no_bias -----------
    rpe = RelativePositionEncoding(r_max=32, s_max=2, c_z=384).eval()
    rp_sd = subtree(sd, "relative_position_encoding")
    trp = {"linear_no_bias.weight": torch.tensor(np.array(rp_sd["linear_no_bias.weight"]))}
    missing, unexpected = rpe.load_state_dict(trp, strict=False)
    print(f"# relpos load: missing={list(missing)} unexpected={list(unexpected)}")

    # Build a random but structurally-plausible feature dict: two chains,
    # two entities, repeated residues so same-residue/same-chain paths fire.
    N = 24
    asym = np.array([0] * 12 + [1] * 12, np.int64)
    entity = np.array([0] * 12 + [1] * 12, np.int64)
    sym = np.array([0] * 12 + [0] * 12, np.int64)
    residue = np.concatenate([np.arange(12), np.arange(12)]).astype(np.int64)
    # duplicate a few residue indices within a chain to exercise same-residue
    residue[3] = residue[2]
    token = np.arange(N).astype(np.int64)
    feat_np = dict(asym_id=asym, residue_index=residue, entity_id=entity,
                   sym_id=sym, token_index=token)

    feat_t = {k: torch.tensor(v) for k, v in feat_np.items()}
    with torch.no_grad():
        relp_t = rpe.generate_relp(dict(feat_t))["relp"]         # [N,N,139]
        proj_t = rpe(relp_t)                                     # [N,N,384]

    feat_m = {k: mx.array(v.astype(np.int32)) for k, v in feat_np.items()}
    relp_m = generate_relp(feat_m, r_max=32, s_max=2)
    proj_m = relative_position_encoding(relp_m, rp_sd)
    mx.eval(relp_m, proj_m)

    print(f"# relp shape torch={tuple(relp_t.shape)} mlx={tuple(relp_m.shape)}")
    results["generate_relp"] = rel(relp_t, relp_m)
    results["relp_projection"] = rel(proj_t, proj_m)

    for k, (d, r) in results.items():
        print(f"{k:28s} {d:13.3e} {r:11.3e}")

    worst = max(r for _, r in results.values())
    print(f"\nEMBED-MISC {'PASS' if worst < 1e-3 else 'FAIL'} (worst rel {worst:.2e})")
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
