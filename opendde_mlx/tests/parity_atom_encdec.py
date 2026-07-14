"""Parity test: MLX atom encoder/decoder vs torch, real checkpoint weights.

Run:
  source .../conda.sh && conda activate of3mlx && cd .../OpenDDE && \
  export LAYERNORM_TYPE=torch OPENDDE_FOLDCP_MODE=single && \
  PYTHONPATH=. python opendde_mlx/tests/parity_atom_encdec.py
"""

from __future__ import annotations

import numpy as np
import torch
import mlx.core as mx

from opendde.model.modules.transformer import (
    AtomAttentionEncoder,
    AtomAttentionDecoder,
    rearrange_qk_to_dense_trunk,
)
from opendde_mlx import atom_encdec as M

CKPT = "/Users/tgg/.cache/opendde/checkpoint/opendde.pt"
C_ATOM = 128
C_ATOMPAIR = 16
C_TOKEN = 384
N_BLOCKS = 3
N_HEADS = 4
NQ, NK = 32, 128


def load_subtree_torch(prefix: str) -> dict:
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    out = {}
    for k, v in sd.items():
        k2 = k[len("module."):] if k.startswith("module.") else k
        if k2.startswith(prefix + "."):
            out[k2[len(prefix) + 1:]] = v.float()
    return out


def to_mlx(sd: dict) -> dict:
    return {k: mx.array(v.numpy()) for k, v in sd.items()}


def rel(a: np.ndarray, b: np.ndarray) -> float:
    num = np.max(np.abs(a - b))
    den = np.max(np.abs(b)) + 1e-9
    return float(num / den), float(num)


def report(name, t, m):
    t = np.asarray(t, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    r, mad = rel(m, t)
    print(f"  {name:16s} shape={tuple(t.shape)} max_abs_diff={mad:.3e} rel={r:.3e}")
    return r


def build_windows(ref_pos_np, N_atom):
    ref_pos = torch.tensor(ref_pos_np, dtype=torch.float32)
    ref_space_uid = torch.arange(N_atom) // 5  # arbitrary grouping into "residues"
    q_tr, k_tr, pad_info = rearrange_qk_to_dense_trunk(
        q=[ref_pos, ref_space_uid], k=[ref_pos, ref_space_uid],
        dim_q=[-2, -1], dim_k=[-2, -1], n_queries=NQ, n_keys=NK, compute_mask=True,
    )
    d_lm = q_tr[0][..., None, :] - k_tr[0][..., None, :, :]      # [nb, nq, nk, 3]
    v_lm = (q_tr[1][..., None].int() == k_tr[1][..., None, :].int()).unsqueeze(-1)  # [nb,nq,nk,1]
    mask_trunked = pad_info["mask_trunked"]
    return d_lm.float(), v_lm.float(), mask_trunked


def test_encoder():
    print("=== AtomAttentionEncoder (has_coords=False) ===")
    torch.manual_seed(0)
    np.random.seed(0)
    N_atom = 96  # 3 windows of 32
    N_token = 20
    prefix = "input_embedder.atom_attention_encoder"
    tsd = load_subtree_torch(prefix)

    enc = AtomAttentionEncoder(
        has_coords=False, c_token=C_TOKEN, c_atom=C_ATOM, c_atompair=C_ATOMPAIR,
        n_blocks=N_BLOCKS, n_heads=N_HEADS, n_queries=NQ, n_keys=NK,
    ).eval()
    missing, unexpected = enc.load_state_dict(tsd, strict=False)
    print(f"  load_state_dict: missing={list(missing)} unexpected={list(unexpected)}")

    # inputs
    ref_pos = np.random.randn(N_atom, 3).astype(np.float32)
    ref_charge = np.random.randn(N_atom).astype(np.float32)
    ref_mask = (np.random.rand(N_atom) > 0.1).astype(np.float32)
    ref_element = np.random.randn(N_atom, 128).astype(np.float32)
    ref_atom_name_chars = np.random.randn(N_atom, 4 * 64).astype(np.float32)
    atom_to_token = np.sort(np.random.randint(0, N_token, size=N_atom)).astype(np.int64)
    atom_to_token[0] = 0
    d_lm, v_lm, mask_trunked = build_windows(ref_pos, N_atom)

    with torch.no_grad():
        a_t, q_t, c_t, plm_t = enc(
            atom_to_token_idx=torch.tensor(atom_to_token),
            ref_pos=torch.tensor(ref_pos),
            ref_charge=torch.tensor(ref_charge),
            ref_mask=torch.tensor(ref_mask),
            ref_atom_name_chars=torch.tensor(ref_atom_name_chars),
            ref_element=torch.tensor(ref_element),
            d_lm=d_lm, v_lm=v_lm,
            pad_info={"mask_trunked": mask_trunked},
        )

    p = to_mlx(tsd)
    a_m, q_m, c_m, plm_m = M.atom_attention_encoder(
        atom_to_token_idx=mx.array(atom_to_token),
        ref_pos=mx.array(ref_pos), ref_charge=mx.array(ref_charge),
        ref_mask=mx.array(ref_mask), ref_atom_name_chars=mx.array(ref_atom_name_chars),
        ref_element=mx.array(ref_element),
        d_lm=mx.array(d_lm.numpy()), v_lm=mx.array(v_lm.numpy()),
        mask_trunked=mx.array(mask_trunked.numpy().astype(np.float32)),
        p=p, n_blocks=N_BLOCKS, n_heads=N_HEADS, n_queries=NQ, n_keys=NK,
    )
    mx.eval(a_m, q_m, c_m, plm_m)

    r1 = report("c_l", c_t.numpy(), np.array(c_m))
    r2 = report("p_lm", plm_t.numpy(), np.array(plm_m))
    r3 = report("q_l (transf)", q_t.numpy(), np.array(q_m))
    r4 = report("a (token)", a_t.numpy(), np.array(a_m))
    return max(r1, r2, r3, r4)


def test_decoder():
    print("=== AtomAttentionDecoder ===")
    torch.manual_seed(1)
    np.random.seed(1)
    N_atom = 96
    N_token = 20
    prefix = "diffusion_module.atom_attention_decoder"
    tsd = load_subtree_torch(prefix)

    c_token_diff = 768  # diffusion path uses c_token=768
    dec = AtomAttentionDecoder(
        n_blocks=N_BLOCKS, n_heads=N_HEADS, c_token=c_token_diff, c_atom=C_ATOM,
        c_atompair=C_ATOMPAIR, n_queries=NQ, n_keys=NK,
    ).eval()
    missing, unexpected = dec.load_state_dict(tsd, strict=False)
    print(f"  load_state_dict: missing={list(missing)} unexpected={list(unexpected)}")

    a = np.random.randn(N_token, c_token_diff).astype(np.float32)
    q_skip = np.random.randn(N_atom, C_ATOM).astype(np.float32)
    c_skip = np.random.randn(N_atom, C_ATOM).astype(np.float32)
    p_skip = np.random.randn(N_atom // NQ, NQ, NK, C_ATOMPAIR).astype(np.float32)
    atom_to_token = np.sort(np.random.randint(0, N_token, size=N_atom)).astype(np.int64)

    with torch.no_grad():
        r_t = dec(
            atom_to_token_idx=torch.tensor(atom_to_token),
            a=torch.tensor(a), q_skip=torch.tensor(q_skip),
            c_skip=torch.tensor(c_skip), p_skip=torch.tensor(p_skip),
        )

    p = to_mlx(tsd)
    r_m = M.atom_attention_decoder(
        atom_to_token_idx=mx.array(atom_to_token),
        a=mx.array(a), q_skip=mx.array(q_skip), c_skip=mx.array(c_skip),
        p_skip=mx.array(p_skip), p=p,
        n_blocks=N_BLOCKS, n_heads=N_HEADS, n_queries=NQ, n_keys=NK,
    )
    mx.eval(r_m)
    return report("r (coords)", r_t.numpy(), np.array(r_m))


if __name__ == "__main__":
    re = test_encoder()
    rd = test_decoder()
    print()
    print(f"WORST encoder rel = {re:.3e}")
    print(f"decoder rel       = {rd:.3e}")
    ok = max(re, rd) < 1e-3
    print("PASS" if ok else "FAIL")
