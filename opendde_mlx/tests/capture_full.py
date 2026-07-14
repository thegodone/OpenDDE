"""Comprehensive capture: FULL featurization (dummy MSA so msa_module runs) +
record the exact sampler noise sequence, for an exact-coordinate MLX match and
proper (~1A) folds. Saves ref_ubiquitin_full.npz."""
import os
os.environ.setdefault("LAYERNORM_TYPE", "torch")
os.environ.setdefault("OPENDDE_FOLDCP_MODE", "single")
os.environ.setdefault("OPENDDE_ROOT_DIR", os.path.expanduser("~/.cache/opendde"))

import numpy as np
import torch

from opendde.config.inference import build_inference_config
from opendde.data.inference.json_to_feature import SampleDictToFeatures
from opendde.data.utils import make_dummy_feature
import opendde.model.utils as mutils
from opendde.model.opendde import OpenDDE

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
CKPT = os.path.expanduser("~/.cache/opendde/checkpoint/opendde.pt")
OUT = os.path.join(os.path.dirname(__file__), "ref_ubiquitin_full.npz")


def main():
    torch.manual_seed(0)
    cfg = build_inference_config(
        arg_str="--use_msa false --use_template false --use_rna_msa false "
                "--triangle_attention torch --triangle_multiplicative torch --dtype fp32 "
                "--sample_diffusion.N_sample 1 --sample_diffusion.N_step 20 --model.N_cycle 4",
        model_name="opendde_v1",
    )
    feats, atom_array, token_array = SampleDictToFeatures(
        {"name": "ubq", "sequences": [{"proteinChain": {"sequence": UBQ, "count": 1}}]}
    ).get_feature_dict()
    make_dummy_feature(feats, dummy_feats=("msa",))     # adds msa/profile/deletion_mean/has_deletion/deletion_value
    print(f"featurized: {len(token_array)} tok, {len(atom_array)} atoms, N_msa={feats['msa'].shape[0]}")

    model = OpenDDE(cfg).eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith("module.")}, strict=False)

    saved = {}

    # --- record the exact noise sequence (init randn, per-step rot/trans/step_noise) ---
    randn_log = []
    _orig_randn = torch.randn
    def rec_randn(*a, **k):
        out = _orig_randn(*a, **k)
        randn_log.append(out.detach().float().cpu().numpy())
        return out
    rot_log = []
    _orig_urr = mutils.uniform_random_rotation
    def rec_urr(*a, **k):
        out = _orig_urr(*a, **k)
        rot_log.append(out.detach().float().cpu().numpy())
        return out
    torch.randn = rec_randn
    mutils.uniform_random_rotation = rec_urr

    # --- capture trunk / structural / denoiser (pre-hooks for in-place) ---
    orig_gp = model.get_pairformer_output
    def wrap_gp(*a, **k):
        si, s, z = orig_gp(*a, **k)
        saved["trunk.s_inputs"] = si.detach().float().cpu().numpy()
        saved["trunk.s"] = s.detach().float().cpu().numpy()
        saved["trunk.z"] = z.detach().float().cpu().numpy()
        return si, s, z
    model.get_pairformer_output = wrap_gp

    orig_exp = model.expand_to_structural_tokens
    def wrap_exp(*a, **k):
        sfd, s2, s3, z3 = orig_exp(*a, **k)
        saved["struct.s_inputs"] = s2.detach().float().cpu().numpy()
        saved["struct.s"] = s3.detach().float().cpu().numpy()
        saved["struct.z"] = z3.detach().float().cpu().numpy()
        for key in ("relp", "ref_pos", "ref_charge", "ref_mask", "ref_atom_name_chars",
                    "ref_element", "d_lm", "v_lm"):
            if key in sfd and torch.is_tensor(sfd[key]):
                saved[f"structfeat.{key}"] = sfd[key].detach().float().cpu().numpy()
        pi = sfd.get("pad_info")
        if isinstance(pi, dict) and torch.is_tensor(pi.get("mask_trunked")):
            saved["structfeat.mask_trunked"] = pi["mask_trunked"].detach().float().cpu().numpy()
        saved["structfeat.atom_to_token_idx"] = sfd["atom_to_token_idx"].detach().float().cpu().numpy()
        return sfd, s2, s3, z3
    model.expand_to_structural_tokens = wrap_exp

    # expander input feats
    for key in ("parent_residue_idx", "subtoken_role_id", "asym_id",
                "prev_parent_residue_idx", "next_parent_residue_idx"):
        if key in feats and torch.is_tensor(feats[key]):
            saved[f"expfeat.{key}"] = feats[key].detach().float().cpu().numpy()

    with torch.no_grad():
        out = model(input_feature_dict=feats, mode="inference")
    torch.randn = _orig_randn
    mutils.uniform_random_rotation = _orig_urr

    pred = out[0] if isinstance(out, (tuple, list)) else out
    saved["final.coordinate"] = pred["coordinate"].detach().float().cpu().numpy()
    print(f"final coords {tuple(pred['coordinate'].shape)}; randn draws={len(randn_log)}, rot draws={len(rot_log)}")

    # store noise sequence
    for i, r in enumerate(randn_log):
        saved[f"randn.{i}"] = r
    for i, r in enumerate(rot_log):
        saved[f"rot.{i}"] = r
    saved["n_randn"] = np.array(len(randn_log))
    saved["n_rot"] = np.array(len(rot_log))
    np.savez(OUT, **saved)
    print(f"saved {len(saved)} arrays -> {OUT}")


if __name__ == "__main__":
    main()
