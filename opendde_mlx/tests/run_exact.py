"""Exact-coordinate match: MLX structural + sampler with torch's INJECTED noise
sequence -> should reproduce torch coordinates atom-for-atom."""
import os
import numpy as np
import mlx.core as mx

from opendde_mlx import structural as ST, diffusion as DF, sampler as SP
from opendde_mlx.weights import load_state_dict, subtree, remap_pairformer_stack

REF = os.path.join(os.path.dirname(__file__), "ref_ubiquitin_full.npz")
CKPT = os.path.expanduser("~/.cache/opendde/checkpoint/opendde.pt")


def main():
    mx.set_default_device(mx.gpu)
    d = np.load(REF); sd = load_state_dict(CKPT)
    def A(k): return mx.array(d[k])

    # --- structural (MLX) on captured (verified) trunk outputs ---
    feats = {k: A(f"expfeat.{k}") for k in ("parent_residue_idx", "subtoken_role_id", "asym_id", "prev_parent_residue_idx", "next_parent_residue_idx")}
    si, s, z, pf = ST.structural_token_expander(subtree(sd, "structural_token_expander"), feats, A("trunk.s_inputs"), A("trunk.s"), A("trunk.z"))
    rp, nb = remap_pairformer_stack(sd, "structural_token_refiner")
    s, z = ST.structural_token_refiner(s, z, rp, n_blocks=nb, extra_attn_bias=pf["structural_pair_attn_bias"])

    # --- diffusion sampler with INJECTED torch noise ---
    ifd = {k: A(f"structfeat.{k}") for k in ("relp", "ref_pos", "ref_charge", "ref_mask", "ref_atom_name_chars", "ref_element", "d_lm", "v_lm")}
    ifd["mask_trunked"] = A("structfeat.mask_trunked")
    ifd["atom_to_token_idx"] = A("structfeat.atom_to_token_idx").astype(mx.int32)
    ifd["structural_pair_attn_bias"] = pf["structural_pair_attn_bias"]
    pdif = subtree(sd, "diffusion_module")
    N_atom = ifd["ref_pos"].shape[0]

    ctx = dict(input_feature_dict=ifd, s_inputs=si, s_trunk=s, z_trunk=z)
    def denoise(x_noisy, t_hat, **kw):
        return DF.diffusion_module_forward(pdif, x_noisy, t_hat, kw["input_feature_dict"], kw["s_inputs"], kw["s_trunk"], kw["z_trunk"], sigma_data=16.0)

    n_step = int(d["n_rot"])                     # 20
    init_noise = A("randn.0")                     # (1, N_atom, 3)
    rots = [A(f"rot.{i}") for i in range(n_step)]                 # (1,3,3) each
    trans = [mx.reshape(A(f"randn.{1 + 2 * i}"), (1, 3)) for i in range(n_step)]   # (1,1,3)->(1,3)
    step_noise = [A(f"randn.{2 + 2 * i}") for i in range(n_step)]                  # (1,N_atom,3)

    sched = SP.noise_schedule(N_step=n_step, sigma_data=16.0)
    coords = SP.sample_diffusion(denoise, sched, (1, N_atom, 3), gamma0=0.8, gamma_min=1.0,
                                 noise_scale_lambda=1.003, step_scale_eta=1.5,
                                 injected_init_noise=init_noise, injected_rots=rots,
                                 injected_trans=trans, injected_step_noise=step_noise,
                                 denoise_context=ctx)
    mx.eval(coords)
    c = np.array(coords); ref = d["final.coordinate"]
    # exact match (same noise) - direct diff, no alignment
    dd = np.abs(c - ref).max(); rms = np.sqrt(((c - ref) ** 2).sum(-1).mean())
    print(f"EXACT-MATCH (injected torch noise): max_abs_diff={dd:.3e} A   direct-RMSD={rms:.4f} A")
    np.save(os.path.join(os.path.dirname(__file__), "odde_mlx_exact.npy"), c)
    print("EXACT-COORD", "PASS" if rms < 0.05 else "FAIL", f"(threshold 0.05 A)")
    return 0 if rms < 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
