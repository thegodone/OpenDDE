"""Speed: OpenDDE (c_z=384) vs OpenFold3 (c_z=128) Pairformer trunk, mx.compile.

The Pairformer trunk is the dominant, shared compute. Both use the identical
block; they differ only in pair width (c_z) and head count -- this is the
'~3x params / ~9x compute' claim, measured on Apple Silicon.
"""
import time
import numpy as np
import torch
import mlx.core as mx

from opendde_mlx import modules as M
from opendde_mlx.weights import subtree, remap_pairformer_stack

OF3 = "/Users/tgg/.openfold3/of3_ft3_v1.pt"
ODDE = "/Users/tgg/.cache/opendde/checkpoint/opendde.pt"


def to_mlx_sd(path, key_get, strip_module):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj[key_get] if (isinstance(obj, dict) and key_get in obj) else obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    out = {}
    for k, v in sd.items():
        if not hasattr(v, "detach"):
            continue
        if strip_module and k.startswith("module."):
            k = k[len("module."):]
        out[k] = mx.array(v.detach().to(torch.float32).numpy())
    return out


def bench(fn, warm=2, reps=6):
    for _ in range(warm):
        mx.eval(*fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval(*fn()); ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1000


def main():
    mx.set_default_device(mx.gpu)

    of3 = to_mlx_sd(OF3, "state_dict", strip_module=False)
    p_of3 = subtree(of3, "pairformer_stack")            # grouped layout, direct
    nb_of3 = 1 + max(int(k.split("blocks.")[1].split(".")[0]) for k in p_of3 if k.startswith("blocks."))

    odde = to_mlx_sd(ODDE, "model", strip_module=True)
    p_odde, nb_odde = remap_pairformer_stack(odde, "pairformer_stack")

    print(f"OF3 : {nb_of3} blocks, c_z=128, heads_pair=4")
    print(f"ODDE: {nb_odde} blocks, c_z=384, heads_pair=12\n")

    rng = np.random.default_rng(0)
    N_CYCLE_OF3, N_CYCLE_ODDE = 4, 10   # recycles at inference

    hdr = f"{'N':>4} | {'OF3 eager':>10} {'OF3 comp':>9} {'cmp x':>6} | {'ODDE eager':>11} {'ODDE comp':>10} {'cmp x':>6} | {'ODDE/OF3':>9}"
    print("--- single Pairformer-trunk pass (48 blocks) ---")
    print(hdr); print("-" * len(hdr))
    trunk_ms = {}
    for N in (76, 128, 256):
        smask, pmask = mx.ones((N,)), mx.ones((N, N))
        s1 = mx.array(rng.standard_normal((N, 384)).astype(np.float32) * 0.1)
        z1 = mx.array(rng.standard_normal((N, N, 128)).astype(np.float32) * 0.1)
        f_of3 = lambda s, z: M.pairformer_stack(s, z, p_of3, smask, pmask, n_blocks=nb_of3, no_heads_pair_bias=16, no_heads_pair=4)
        e_of3 = bench(lambda: (f_of3(s1, z1),)); c_of3 = mx.compile(f_of3); t_of3 = bench(lambda: (c_of3(s1, z1),))

        s2 = mx.array(rng.standard_normal((N, 384)).astype(np.float32) * 0.1)
        z2 = mx.array(rng.standard_normal((N, N, 384)).astype(np.float32) * 0.1)
        f_odde = lambda s, z: M.pairformer_stack(s, z, p_odde, smask, pmask, n_blocks=nb_odde, no_heads_pair_bias=16, no_heads_pair=12)
        e_odde = bench(lambda: (f_odde(s2, z2),)); c_odde = mx.compile(f_odde); t_odde = bench(lambda: (c_odde(s2, z2),))

        trunk_ms[N] = (t_of3, t_odde)
        print(f"{N:>4} | {e_of3:>8.1f}ms {t_of3:>7.1f}ms {e_of3/t_of3:>5.2f}x | {e_odde:>9.1f}ms {t_odde:>8.1f}ms {e_odde/t_odde:>5.2f}x | {t_odde/t_of3:>8.2f}x")

    print("\n--- projected full-trunk cost (trunk x recycles, compiled) ---")
    print(f"{'N':>4} | {'OF3 x4':>10} {'ODDE x10':>10} {'ODDE/OF3':>9}")
    for N in (76, 128, 256):
        to, td = trunk_ms[N]
        fo, fd = to * N_CYCLE_OF3, td * N_CYCLE_ODDE
        print(f"{N:>4} | {fo/1000:>8.2f}s {fd/1000:>8.2f}s {fd/fo:>8.2f}x")

    print("\nOF3: c_z=128 heads=4, N_cycle=4.  ODDE: c_z=384 heads=12, N_cycle=10.")
    print("Single-pass ~4-4.7x (pair width 3x -> triangle ops ~9x, tempered by")
    print("attn-pair-bias/transition that scale with c_s=384, shared). With 2.5x more")
    print("recycles, ODDE's trunk wall-clock is ~10-12x OF3's on the same M3 Max GPU.")


if __name__ == "__main__":
    main()
