"""Run the torch OpenDDE on ubiquitin, capture trunk I/O + feature dict for the
MLX e2e gate. Saves everything to opendde_mlx/tests/ref_ubiquitin.npz."""
import os
os.environ.setdefault("LAYERNORM_TYPE", "torch")
os.environ.setdefault("OPENDDE_FOLDCP_MODE", "single")
os.environ.setdefault("OPENDDE_ROOT_DIR", os.path.expanduser("~/.cache/opendde"))

import numpy as np
import torch

from opendde.config.inference import build_inference_config
from opendde.data.inference.json_to_feature import SampleDictToFeatures
from opendde.model.opendde import OpenDDE

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
CKPT = os.path.expanduser("~/.cache/opendde/checkpoint/opendde.pt")
OUT = os.path.join(os.path.dirname(__file__), "ref_ubiquitin.npz")


def main():
    torch.manual_seed(0); np.random.seed(0)
    cfg = build_inference_config(
        arg_str="--use_msa false --use_template false --use_rna_msa false "
                "--triangle_attention torch --triangle_multiplicative torch --dtype fp32 "
                "--sample_diffusion.N_sample 1 --sample_diffusion.N_step 20 --model.N_cycle 4",
        model_name="opendde_v1",
    )
    print("config built; N_cycle=", cfg.model.N_cycle, "N_step=", cfg.sample_diffusion.N_step)

    feats, atom_array, token_array = SampleDictToFeatures(
        {"name": "ubq", "sequences": [{"proteinChain": {"sequence": UBQ, "count": 1}}]}
    ).get_feature_dict()
    print(f"featurized ubiquitin: {len(token_array)} tokens, {len(atom_array)} atoms")
    # dummy-MSA features (no-MSA path, opendde/data/utils.py:820)
    feats["profile"] = feats["restype"].clone().float()
    feats["deletion_mean"] = torch.zeros(feats["restype"].shape[0], dtype=torch.float32)

    model = OpenDDE(cfg).eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    sd = {k[len("module."):]: v for k, v in sd.items() if k.startswith("module.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"model load: missing={len(missing)} unexpected={len(unexpected)}")

    saved = {}

    # PRE-hook (before forward) + CLONE: the pairformer mutates its input in place,
    # so a post-hook would capture the mutated tensor, not the true input.
    def pf_pre(mod, args, kwargs):
        s_in = args[0] if len(args) > 0 else kwargs.get("s")
        z_in = args[1] if len(args) > 1 else kwargs.get("z")
        if torch.is_tensor(s_in):
            saved["pf_in.s"] = s_in.detach().clone().float().cpu().numpy()
        if torch.is_tensor(z_in):
            saved["pf_in.z"] = z_in.detach().clone().float().cpu().numpy()
        print(f"  [pre-hook] pf input cloned: z{tuple(z_in.shape)}")
    model.pairformer_stack.register_forward_pre_hook(pf_pre, with_kwargs=True)

    # pre-hook msa_module to capture z entering it (= z_init + z_cycle, before msa)
    def msa_pre(mod, args, kwargs):
        # msa_module.forward(input_feature_dict, z, s_inputs, ...) -> z is arg[1]
        fd = args[0] if len(args) > 0 else kwargs.get("input_feature_dict")
        z_in = args[1] if len(args) > 1 else kwargs.get("z")
        if torch.is_tensor(z_in):
            saved["msa_in.z"] = z_in.detach().clone().float().cpu().numpy()
        for key in ("msa", "has_deletion", "deletion_value", "msa_mask", "profile"):
            if isinstance(fd, dict) and key in fd and torch.is_tensor(fd[key]):
                saved[f"msafeat.{key}"] = fd[key].detach().clone().float().cpu().numpy()
        print(f"  [pre-hook] msa input z + feats cloned; msa keys: "
              f"{[k for k in ('msa','has_deletion','deletion_value','profile') if isinstance(fd,dict) and k in fd]}")
    model.msa_module.register_forward_pre_hook(msa_pre, with_kwargs=True)

    def pf_post(mod, args, kwargs, out):
        if isinstance(out, (tuple, list)) and torch.is_tensor(out[1]):
            saved["pf_out.s"] = out[0].detach().clone().float().cpu().numpy()
            saved["pf_out.z"] = out[1].detach().clone().float().cpu().numpy()
    model.pairformer_stack.register_forward_hook(pf_post, with_kwargs=True)

    orig = model.get_pairformer_output
    def wrapped(input_feature_dict, *a, **k):
        # snapshot the feature fields the trunk consumes
        for key in ("restype", "profile", "deletion_mean", "relp", "token_bonds",
                    "atom_to_token_idx", "ref_pos", "ref_charge", "ref_mask",
                    "ref_atom_name_chars", "ref_element", "d_lm", "v_lm",
                    "msa", "has_deletion", "deletion_value"):
            if key in input_feature_dict and torch.is_tensor(input_feature_dict[key]):
                saved[f"feat.{key}"] = input_feature_dict[key].detach().float().cpu().numpy()
        if "pad_info" in input_feature_dict:
            pi = input_feature_dict["pad_info"]
            if isinstance(pi, dict):
                for kk, vv in pi.items():
                    if torch.is_tensor(vv):
                        saved[f"padinfo.{kk}"] = vv.detach().float().cpu().numpy()
        s_inputs, s, z = orig(input_feature_dict, *a, **k)
        saved["trunk.s_inputs"] = s_inputs.detach().float().cpu().numpy()
        saved["trunk.s"] = s.detach().float().cpu().numpy()
        saved["trunk.z"] = z.detach().float().cpu().numpy()
        print(f"captured trunk: s_inputs{tuple(s_inputs.shape)} s{tuple(s.shape)} z{tuple(z.shape)}")
        return s_inputs, s, z   # continue to structural + diffusion
    model.get_pairformer_output = wrapped

    # capture structural expander output
    orig_exp = model.expand_to_structural_tokens
    def wrapped_exp(*a, **k):
        ifd = a[0] if a else k.get("input_feature_dict")
        for key in ("parent_residue_idx", "subtoken_role_id", "asym_id",
                    "prev_parent_residue_idx", "next_parent_residue_idx"):
            if isinstance(ifd, dict) and key in ifd and torch.is_tensor(ifd[key]):
                saved[f"expfeat.{key}"] = ifd[key].detach().float().cpu().numpy()
        sfd, s2, s3, z3 = orig_exp(*a, **k)
        saved["struct.s_inputs"] = s2.detach().float().cpu().numpy()
        saved["struct.s"] = s3.detach().float().cpu().numpy()
        saved["struct.z"] = z3.detach().float().cpu().numpy()
        for key in ("relp", "atom_to_token_idx", "ref_pos", "ref_charge", "ref_mask",
                    "ref_atom_name_chars", "ref_element", "d_lm", "v_lm"):
            if key in sfd and torch.is_tensor(sfd[key]):
                saved[f"structfeat.{key}"] = sfd[key].detach().float().cpu().numpy()
        pi = sfd.get("pad_info")
        if isinstance(pi, dict) and torch.is_tensor(pi.get("mask_trunked")):
            saved["structfeat.mask_trunked"] = pi["mask_trunked"].detach().float().cpu().numpy()
        print(f"captured structural: s{tuple(s3.shape)} z{tuple(z3.shape)}")
        return sfd, s2, s3, z3
    model.expand_to_structural_tokens = wrapped_exp

    # capture diffusion denoiser first-call I/O
    diff_seen = {"n": 0}
    def diff_pre(mod, args, kwargs):
        if diff_seen["n"] == 0:
            xn = args[0] if len(args) > 0 else kwargs.get("x_noisy")
            th = args[1] if len(args) > 1 else kwargs.get("t_hat_noise_level", kwargs.get("t_hat"))
            if torch.is_tensor(xn): saved["denoise.x_noisy"] = xn.detach().clone().float().cpu().numpy()
            if torch.is_tensor(th): saved["denoise.t_hat"] = th.detach().clone().float().cpu().numpy()
    def diff_post(mod, args, kwargs, out):
        if diff_seen["n"] == 0 and torch.is_tensor(out):
            saved["denoise.x_denoised"] = out.detach().clone().float().cpu().numpy()
            print(f"captured denoiser: x_noisy->x_denoised {tuple(out.shape)}")
        diff_seen["n"] += 1
        # also always overwrite the "last" call
        if torch.is_tensor(out):
            saved["denoiseL.x_denoised"] = out.detach().clone().float().cpu().numpy()
    def diff_pre_last(mod, args, kwargs):
        xn = args[0] if len(args) > 0 else kwargs.get("x_noisy")
        th = args[1] if len(args) > 1 else kwargs.get("t_hat_noise_level", kwargs.get("t_hat"))
        if torch.is_tensor(xn): saved["denoiseL.x_noisy"] = xn.detach().clone().float().cpu().numpy()
        if torch.is_tensor(th): saved["denoiseL.t_hat"] = th.detach().clone().float().cpu().numpy()
    model.diffusion_module.register_forward_pre_hook(diff_pre_last, with_kwargs=True)
    model.diffusion_module.register_forward_pre_hook(diff_pre, with_kwargs=True)
    model.diffusion_module.register_forward_hook(diff_post, with_kwargs=True)

    # hook diffusion sub-modules to localize (first call only)
    def sub_hook(tag):
        def h(mod, args, kwargs, out):
            if diff_seen["n"] != 0:
                return
            if torch.is_tensor(out):
                saved[f"diffsub.{tag}"] = out.detach().clone().float().cpu().numpy()
            elif isinstance(out, (tuple, list)):
                for i, o in enumerate(out):
                    if torch.is_tensor(o):
                        saved[f"diffsub.{tag}.{i}"] = o.detach().clone().float().cpu().numpy()
        return h
    dm = model.diffusion_module
    def atomenc_pre(mod, args, kwargs):
        if diff_seen["n"] != 0:
            return
        for name in ("r_l", "s", "z"):
            v = kwargs.get(name)
            if torch.is_tensor(v):
                saved[f"atomenc_in.{name}"] = v.detach().clone().float().cpu().numpy()
        # also positional fallback: signature (..., r_l, s, z, ...)
        print(f"  [atomenc input] kwargs: {[k for k in kwargs if torch.is_tensor(kwargs[k])]}")
    dm.atom_attention_encoder.register_forward_pre_hook(atomenc_pre, with_kwargs=True)
    def tr_pre(mod, args, kwargs):
        if diff_seen["n"] != 0: return
        a_in = args[0] if args else kwargs.get("a", kwargs.get("token_repr"))
        if torch.is_tensor(a_in):
            saved["tr_in.a"] = a_in.detach().clone().float().cpu().numpy()
        print(f"  [transformer input] args={len(args)} kw={[k for k in kwargs if torch.is_tensor(kwargs[k])]}")
    dm.diffusion_transformer.register_forward_pre_hook(tr_pre, with_kwargs=True)
    dm.diffusion_conditioning.register_forward_hook(sub_hook("cond"), with_kwargs=True)
    dm.atom_attention_encoder.register_forward_hook(sub_hook("atomenc"), with_kwargs=True)
    dm.diffusion_transformer.register_forward_hook(sub_hook("transformer"), with_kwargs=True)
    dm.atom_attention_decoder.register_forward_hook(sub_hook("atomdec"), with_kwargs=True)

    with torch.no_grad():
        out = model(input_feature_dict=feats, mode="inference")
    pred = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(pred, dict) and torch.is_tensor(pred.get("coordinate")):
        saved["final.coordinate"] = pred["coordinate"].detach().float().cpu().numpy()
        print(f"captured final coords: {tuple(pred['coordinate'].shape)}")

    np.savez(OUT, **saved)
    print(f"saved {len(saved)} arrays -> {OUT}")


if __name__ == "__main__":
    main()
