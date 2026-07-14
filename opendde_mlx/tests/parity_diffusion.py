"""torch<->MLX parity for the OpenDDE DiffusionModule (Alg 20) + DiffusionConditioning
(Alg 21, active pair compression c_z 384 -> c_z_pair_diffusion 128), with real weights.

Builds the torch DiffusionModule(c_z=384, c_z_pair_diffusion=128), loads the real
`diffusion_module` checkpoint subtree, constructs a small fixed random input
(identical arrays fed to both paths), and compares:
  * DiffusionConditioning -> (single_s, pair_z)
  * AtomAttentionEncoder  -> (a_token, q_skip, c_skip, p_skip)
  * full DiffusionModule.forward -> x_denoised
"""
import os
os.environ.setdefault("LAYERNORM_TYPE", "torch")
os.environ.setdefault("OPENDDE_FOLDCP_MODE", "single")

import numpy as np
import torch
import mlx.core as mx

from opendde.model.modules.diffusion import DiffusionModule
from opendde.model.modules.primitives import rearrange_qk_to_dense_trunk as t_rearrange
from opendde_mlx import diffusion as D
from opendde_mlx.weights import load_state_dict, subtree

CKPT = "/Users/tgg/.cache/opendde/checkpoint/opendde.pt"

C_S, C_S_INPUTS, C_Z = 384, 449, 384
SIGMA_DATA = 16.0
RMAX, SMAX = 32, 2
RELP_DIM = 4 * RMAX + 2 * SMAX + 7  # 139


def raw_torch_sd():
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = obj["model"]
    return {k[len("module."):]: v for k, v in sd.items() if k.startswith("module.")}


def t2m(t):
    return mx.array(t.detach().to(torch.float32).numpy())


def rel(a_t, b_m):
    a = np.asarray(a_t.detach().to(torch.float32).numpy())
    b = np.array(b_m, copy=False)
    d = np.abs(a - b).max()
    r = d / (np.abs(a).max() + 1e-9)
    return float(d), float(r)


def main():
    mx.set_default_device(mx.gpu)
    torch.manual_seed(0)
    rng = np.random.default_rng(1)

    N_token, N_atom, N_sample = 8, 40, 2
    n_queries, n_keys = 32, 128

    # --- torch module + real weights ---
    module = DiffusionModule(c_z=C_Z, c_z_pair_diffusion=128).eval()
    raw = raw_torch_sd()
    dm_sd = {k[len("diffusion_module."):]: v for k, v in raw.items()
             if k.startswith("diffusion_module.")}
    missing, unexpected = module.load_state_dict(dm_sd, strict=False)
    missing = [m for m in missing if "foldcp" not in m.lower()]
    print(f"torch load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing sample:", missing[:5])
    if unexpected:
        print("  unexpected sample:", unexpected[:5])

    mlx_sd = load_state_dict(CKPT)
    p = subtree(mlx_sd, "diffusion_module")

    # --- fixed random inputs (numpy) fed to both paths ---
    def R(*shape, scale=1.0):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    x_noisy = R(N_sample, N_atom, 3, scale=5.0)
    t_hat = (np.abs(rng.standard_normal((N_sample,))) * 8.0 + 0.5).astype(np.float32)
    s_inputs = R(N_token, C_S_INPUTS, scale=0.5)
    s_trunk = R(N_token, C_S, scale=0.5)
    z_trunk = R(N_token, N_token, C_Z, scale=0.3)
    relp = R(N_token, N_token, RELP_DIM, scale=0.5)

    # atoms grouped contiguously into tokens
    atom_to_token = np.sort(rng.integers(0, N_token, size=(N_atom,))).astype(np.int64)
    atom_to_token[0] = 0
    atom_to_token[-1] = N_token - 1

    ref_pos = R(N_atom, 3)
    ref_charge = R(N_atom)
    ref_mask = np.ones((N_atom,), np.float32)
    ref_element = R(N_atom, 128, scale=0.5)
    ref_atom_name_chars = R(N_atom, 4, 64, scale=0.5)

    # atom-pair trunk features + pad_info (mask_trunked) from torch rearrange
    atk_t = torch.tensor(atom_to_token)
    _, _, pad_info = t_rearrange(atk_t, atk_t, dim_q=-1, dim_k=-1,
                                 n_queries=n_queries, n_keys=n_keys, compute_mask=True)
    mask_trunked = pad_info["mask_trunked"]  # [n_blocks, n_q, n_k] bool
    n_blocks = mask_trunked.shape[-3]
    d_lm = R(n_blocks, n_queries, n_keys, 3, scale=0.5)
    v_lm = np.abs(R(n_blocks, n_queries, n_keys, 1)).astype(np.float32)
    mask_trunked_np = mask_trunked.to(torch.float32).numpy()

    # torch tensors
    tt = lambda a: torch.tensor(a)
    ifd_t = {
        "relp": tt(relp),
        "atom_to_token_idx": atk_t,
        "ref_pos": tt(ref_pos),
        "ref_charge": tt(ref_charge),
        "ref_mask": tt(ref_mask),
        "ref_atom_name_chars": tt(ref_atom_name_chars),
        "ref_element": tt(ref_element),
        "d_lm": tt(d_lm),
        "v_lm": tt(v_lm),
        "pad_info": pad_info,
    }
    # mlx feature dict
    ifd_m = {
        "relp": mx.array(relp),
        "atom_to_token_idx": mx.array(atom_to_token.astype(np.int32)),
        "ref_pos": mx.array(ref_pos),
        "ref_charge": mx.array(ref_charge),
        "ref_mask": mx.array(ref_mask),
        "ref_atom_name_chars": mx.array(ref_atom_name_chars),
        "ref_element": mx.array(ref_element),
        "d_lm": mx.array(d_lm),
        "v_lm": mx.array(v_lm),
        "mask_trunked": mx.array(mask_trunked_np),
    }

    results = {}
    with torch.no_grad():
        # 1) DiffusionConditioning
        s_single_t, z_pair_t = module.diffusion_conditioning(
            tt(t_hat), tt(relp), s_inputs=tt(s_inputs), s_trunk=tt(s_trunk),
            z_trunk=tt(z_trunk), pair_z=None,
        )
        s_single_m, z_pair_m = D.diffusion_conditioning(
            subtree(p, "diffusion_conditioning"),
            mx.array(t_hat), mx.array(relp), mx.array(s_inputs),
            mx.array(s_trunk), mx.array(z_trunk), sigma_data=SIGMA_DATA,
        )
        mx.eval(s_single_m, z_pair_m)
        results["cond.single_s"] = rel(s_single_t, s_single_m)
        results["cond.pair_z"] = rel(z_pair_t, z_pair_m)

        # 2) AtomAttentionEncoder (as called inside f_forward)
        r_noisy_t = tt(x_noisy) / torch.sqrt(
            torch.tensor(SIGMA_DATA) ** 2 + tt(t_hat) ** 2
        )[..., None, None]
        s_enc_t = s_single_t.new_tensor(s_trunk).unsqueeze(-3)  # [..., 1, N, c_s]
        a_tok_t, q_skip_t, c_skip_t, p_skip_t = module.atom_attention_encoder(
            ifd_t["atom_to_token_idx"], ifd_t["ref_pos"], ifd_t["ref_charge"],
            ifd_t["ref_mask"], ifd_t["ref_atom_name_chars"], ifd_t["ref_element"],
            ifd_t["d_lm"], ifd_t["v_lm"], ifd_t["pad_info"],
            r_l=r_noisy_t, s=s_enc_t, z=z_pair_t,
        )
        r_noisy_m = mx.array(x_noisy) / mx.sqrt(SIGMA_DATA ** 2 + mx.array(t_hat) ** 2)[..., None, None]
        a_tok_m, q_skip_m, c_skip_m, p_skip_m = D.atom_attention_encoder(
            subtree(p, "atom_attention_encoder"),
            ifd_m["atom_to_token_idx"], ifd_m["ref_pos"], ifd_m["ref_charge"],
            ifd_m["ref_mask"], ifd_m["ref_atom_name_chars"], ifd_m["ref_element"],
            ifd_m["d_lm"], ifd_m["v_lm"], ifd_m["mask_trunked"],
            r_noisy_m, mx.array(s_trunk)[..., None, :, :], z_pair_m,
        )
        mx.eval(a_tok_m, q_skip_m, c_skip_m, p_skip_m)
        results["enc.a_token"] = rel(a_tok_t, a_tok_m)
        results["enc.q_skip"] = rel(q_skip_t, q_skip_m)
        results["enc.c_skip"] = rel(c_skip_t, c_skip_m)
        results["enc.p_skip"] = rel(p_skip_t, p_skip_m)

        # 3) full DiffusionModule.forward
        x_den_t = module.forward(
            tt(x_noisy), tt(t_hat), ifd_t, tt(s_inputs), tt(s_trunk),
            tt(z_trunk), None, None, None,
        )
    x_den_m = D.diffusion_module_forward(
        p, mx.array(x_noisy), mx.array(t_hat), ifd_m, mx.array(s_inputs),
        mx.array(s_trunk), mx.array(z_trunk), sigma_data=SIGMA_DATA,
    )
    mx.eval(x_den_m)
    results["MODULE.x_denoised"] = rel(x_den_t, x_den_m)

    print(f"\n{'step':24s} {'max_abs_diff':>13s} {'rel':>11s}")
    for k, (d, r) in results.items():
        print(f"{k:24s} {d:13.3e} {r:11.3e}")

    worst = max(r for _, r in results.values())
    ok = worst < 1e-3
    print(f"\nDIFFUSION-PARITY {'PASS' if ok else 'FAIL'} (worst rel {worst:.2e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
