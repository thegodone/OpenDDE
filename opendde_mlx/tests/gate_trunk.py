"""S6: full MLX trunk (z_init + 1-cycle recycling + msa + pairformer) on ubiquitin,
gated vs captured torch trunk.s/z. Uses verified s_inputs as the entry."""
import os
import numpy as np
import mlx.core as mx

from opendde_mlx import modules as M
from opendde_mlx.modules import linear, layer_norm, _sub
from opendde_mlx.embed_misc import relative_position_encoding
from opendde_mlx.msa import project_msa_sample, msa_module_blocks
from opendde_mlx.weights import load_state_dict, subtree, remap_pairformer_stack

REF = os.path.join(os.path.dirname(__file__), "ref_ubiquitin.npz")
CKPT = os.path.expanduser("~/.cache/opendde/checkpoint/opendde.pt")


def main():
    mx.set_default_device(mx.gpu)
    d = np.load(REF)
    sd = load_state_dict(CKPT)

    s_inputs = mx.array(d["trunk.s_inputs"][0])   # verified 1e-7
    relp = mx.array(d["feat.relp"][0])
    token_bonds = mx.array(d["feat.token_bonds"][0])
    restype = mx.array(d["feat.restype"][0])       # [N,32] one-hot
    N = s_inputs.shape[0]

    # --- s_init, z_init ---
    s_init = linear(s_inputs, sd["linear_no_bias_sinit.weight"])          # [N,384]
    z_init = (linear(s_init, sd["linear_no_bias_zinit1.weight"])[:, None, :]
              + linear(s_init, sd["linear_no_bias_zinit2.weight"])[None, :, :])
    z_init = z_init + relative_position_encoding(relp, {"linear_no_bias.weight": sd["relative_position_encoding.linear_no_bias.weight"]})
    z_init = z_init + linear(token_bonds[..., None], sd["linear_no_bias_token_bond.weight"])

    # --- recycling (N_cycle=1): z,s start at 0 ---
    z = mx.zeros((N, N, 384)); s = mx.zeros((N, 384))
    z = z_init + linear(layer_norm(z, sd["layernorm_z_cycle.weight"], sd["layernorm_z_cycle.bias"]),
                        sd["linear_no_bias_z_cycle.weight"])
    # template = 0 (no template).
    # msa_module: no-MSA path -> "msa" not in feature dict -> _prepare_msa_sample
    # returns None -> z returned UNCHANGED. So skip it entirely.
    # --- s recycle + pairformer ---
    s = s_init + linear(layer_norm(s, sd["layernorm_s.weight"], sd["layernorm_s.bias"]),
                        sd["linear_no_bias_s.weight"])
    # localize: pre-pairformer z,s should equal captured pf_in
    mx.eval(s, z)
    for name, ref, got in [("pre-pf s", d["pf_in.s"][0], np.array(s)), ("pre-pf z", d["pf_in.z"][0], np.array(z))]:
        dd = np.abs(ref - got).max(); print(f"  {name}: max_abs={dd:.3e} rel={dd/(np.abs(ref).max()+1e-9):.3e}")
    p_pf, nb = remap_pairformer_stack(sd, "pairformer_stack")
    s, z = M.pairformer_stack(s, z, p_pf, mx.ones((N,)), mx.ones((N, N)), n_blocks=nb,
                              no_heads_pair_bias=16, no_heads_pair=12, inf=1e9)
    mx.eval(s, z)

    for name, ref, got in [("s", d["trunk.s"][0], np.array(s)), ("z", d["trunk.z"][0], np.array(z))]:
        dd = np.abs(ref - got).max(); rr = dd / (np.abs(ref).max() + 1e-9)
        print(f"  full trunk {name}: max_abs={dd:.3e} rel={rr:.3e}")
    # localize: also report the pre-pairformer z (should match pf_in.z)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
