"""S6 integration gate #1: MLX input embedder s_inputs vs torch, on REAL ubiquitin
features captured in ref_ubiquitin.npz."""
import os
import numpy as np
import mlx.core as mx

from opendde_mlx import modules as M
from opendde_mlx.atom_encdec import atom_attention_encoder
from opendde_mlx.weights import load_state_dict, subtree

REF = os.path.join(os.path.dirname(__file__), "ref_ubiquitin.npz")
CKPT = os.path.expanduser("~/.cache/opendde/checkpoint/opendde.pt")


def main():
    mx.set_default_device(mx.gpu)
    d = np.load(REF)
    def f(name): return mx.array(d[f"feat.{name}"][0])   # drop batch dim

    sd = load_state_dict(CKPT)
    p = subtree(sd, "input_embedder.atom_attention_encoder")

    a, q_l, c_l, p_lm = atom_attention_encoder(
        atom_to_token_idx=mx.array(d["feat.atom_to_token_idx"][0].astype(np.int32)),
        ref_pos=f("ref_pos"), ref_charge=f("ref_charge"), ref_mask=f("ref_mask"),
        ref_atom_name_chars=f("ref_atom_name_chars"), ref_element=f("ref_element"),
        d_lm=f("d_lm"), v_lm=f("v_lm"),
        mask_trunked=mx.array(d["padinfo.mask_trunked"][0]),
        p=p,
    )
    # InputFeatureEmbedder concat: [a(384), restype(32), profile(32), deletion_mean(1)] = 449
    dm = f("deletion_mean")
    dm = dm[..., None] if dm.ndim == 1 else dm
    s_inputs = mx.concatenate([a, f("restype"), f("profile"), dm], axis=-1)
    mx.eval(s_inputs)

    ref = d["trunk.s_inputs"][0]
    got = np.array(s_inputs)
    # split the diff by segment to localize any mismatch
    seg = {"a[0:384]": (0, 384), "restype[384:416]": (384, 416),
           "profile[416:448]": (416, 448), "deletion[448:449]": (448, 449)}
    print(f"s_inputs shape torch{ref.shape} mlx{got.shape}")
    for name, (lo, hi) in seg.items():
        dd = np.abs(ref[..., lo:hi] - got[..., lo:hi]).max()
        rr = dd / (np.abs(ref[..., lo:hi]).max() + 1e-9)
        print(f"  {name:22s} max_abs={dd:.3e} rel={rr:.3e}")
    tot = np.abs(ref - got).max(); rel = tot / (np.abs(ref).max() + 1e-9)
    print(f"s_inputs TOTAL max_abs={tot:.3e} rel={rel:.3e}")
    ok = rel < 1e-3
    print("S_INPUTS-GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
