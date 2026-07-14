"""Full ODDE-mlx fold of ubiquitin: MLX trunk -> structural -> diffusion sampler
-> coordinates. Writes coords for comparison with ODDE-torch + 1UBQ.

Neural chain (trunk, structural, denoiser) is all MLX and gated <1e-5 vs torch.
Structural feature-prep (structfeat: structural relp / atom windows) is reused
from the captured torch featurization (deterministic input prep, not learned)."""
import os
import numpy as np
import mlx.core as mx

from opendde_mlx import modules as M, structural as ST, diffusion as DF, sampler as SP
from opendde_mlx.modules import linear, layer_norm
from opendde_mlx.embed_misc import relative_position_encoding
from opendde_mlx.weights import load_state_dict, subtree, remap_pairformer_stack

REF = os.path.join(os.path.dirname(__file__), "ref_ubiquitin.npz")
CKPT = os.path.expanduser("~/.cache/opendde/checkpoint/opendde.pt")


def mlx_trunk(sd, d, n_cycle=4):
    def A(k): return mx.array(d[k])
    s_inputs = A("trunk.s_inputs")              # verified 1e-7 (input embedder)
    relp, tb, N = A("feat.relp"), A("feat.token_bonds"), s_inputs.shape[0]
    s_init = linear(s_inputs, sd["linear_no_bias_sinit.weight"])
    z_init = (linear(s_init, sd["linear_no_bias_zinit1.weight"])[:, None, :]
              + linear(s_init, sd["linear_no_bias_zinit2.weight"])[None, :, :])
    z_init = z_init + relative_position_encoding(relp, {"linear_no_bias.weight": sd["relative_position_encoding.linear_no_bias.weight"]})
    z_init = z_init + linear(tb[..., None], sd["linear_no_bias_token_bond.weight"])
    p_pf, nb = remap_pairformer_stack(sd, "pairformer_stack")
    z = mx.zeros((N, N, 384)); s = mx.zeros((N, 384))
    for _ in range(n_cycle):
        z = z_init + linear(layer_norm(z, sd["layernorm_z_cycle.weight"], sd["layernorm_z_cycle.bias"]), sd["linear_no_bias_z_cycle.weight"])
        # msa/template no-op (use_msa/template=false)
        s = s_init + linear(layer_norm(s, sd["layernorm_s.weight"], sd["layernorm_s.bias"]), sd["linear_no_bias_s.weight"])
        s, z = M.pairformer_stack(s, z, p_pf, mx.ones((N,)), mx.ones((N, N)), n_blocks=nb, no_heads_pair_bias=16, no_heads_pair=12)
        mx.eval(s, z)
    return s_inputs, s, z


def main():
    mx.set_default_device(mx.gpu)
    d = np.load(REF); sd = load_state_dict(CKPT)
    def A(k): return mx.array(d[k])

    # --- trunk (MLX) ---
    s_inputs, s, z = mlx_trunk(sd, d)

    # --- structural (MLX) ---
    feats = {k: A(f"expfeat.{k}") for k in ("parent_residue_idx", "subtoken_role_id", "asym_id", "prev_parent_residue_idx", "next_parent_residue_idx")}
    si_s, s_s, z_s, pf = ST.structural_token_expander(subtree(sd, "structural_token_expander"), feats, s_inputs, s, z)
    ref_p, nb = remap_pairformer_stack(sd, "structural_token_refiner")
    s_s, z_s = ST.structural_token_refiner(s_s, z_s, ref_p, n_blocks=nb, extra_attn_bias=pf["structural_pair_attn_bias"])

    # --- diffusion (MLX): sampler over the fixed denoiser ---
    ifd = {k: A(f"structfeat.{k}") for k in ("relp", "ref_pos", "ref_charge", "ref_mask", "ref_atom_name_chars", "ref_element", "d_lm", "v_lm")}
    ifd["mask_trunked"] = A("structfeat.mask_trunked")
    ifd["atom_to_token_idx"] = A("structfeat.atom_to_token_idx").astype(mx.int32)
    ifd["structural_pair_attn_bias"] = pf["structural_pair_attn_bias"]
    pdif = subtree(sd, "diffusion_module")
    N_atom = ifd["ref_pos"].shape[0]

    ctx = dict(input_feature_dict=ifd, s_inputs=si_s, s_trunk=s_s, z_trunk=z_s)
    def denoise(x_noisy, t_hat, **kw):
        return DF.diffusion_module_forward(pdif, x_noisy, t_hat, kw["input_feature_dict"],
                                           kw["s_inputs"], kw["s_trunk"], kw["z_trunk"], sigma_data=16.0)

    sched = SP.noise_schedule(N_step=20, sigma_data=16.0)
    mx.random.seed(42)
    coords = SP.sample_diffusion(denoise, sched, (1, N_atom, 3), gamma0=0.8, gamma_min=1.0,
                                 noise_scale_lambda=1.003, step_scale_eta=1.5, denoise_context=ctx)
    mx.eval(coords)
    c = np.array(coords)
    print(f"ODDE-mlx coordinates: shape {c.shape} range [{c.min():.2f}, {c.max():.2f}]")
    np.save(os.path.join(os.path.dirname(__file__), "odde_mlx_coords.npy"), c)
    # sanity: intra-structure Ca-Ca distances reasonable? report radius of gyration
    ca = c[0]
    rg = np.sqrt(((ca - ca.mean(0))**2).sum(1).mean())
    print(f"radius of gyration (all atoms): {rg:.2f} A  (ubiquitin ~ 12 A)")
    print("finite:", np.isfinite(c).all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
