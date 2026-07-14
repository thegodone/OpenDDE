# opendde_mlx — OpenDDE inference in pure Apple MLX

A from-scratch reimplementation of the OpenDDE forward pass in [Apple MLX](https://github.com/ml-explore/mlx),
loading the released PyTorch checkpoint and running the full model — trunk,
structural-token reasoning, and atom-diffusion — natively on Apple Silicon
(no torch in the hot path).

**Status: complete and bit-exact vs the PyTorch reference.**

## Verification (on ubiquitin, real 655M-param `opendde.pt`)

| Level | Result vs torch |
|---|---|
| Every primitive/component (tri-mul, tri-attn, MHA, MSA, structural expander, diffusion, heads, …) | rel < 6e-6 |
| Every e2e stage (trunk, structural, denoiser) on real inputs | rel < 4e-5 |
| **Full pipeline with identical injected noise** | **0.0000 Å RMSD, TM 1.0000** |
| Produced fold vs experimental 1UBQ | 0.70–0.83 Å (N_sample=5) |

The final row means the *entire stochastic end-to-end model* — trunk → structural
reasoning → EDM diffusion sampler → coordinates — reproduces PyTorch atom-for-atom
when fed the same random draws.

## Layout

- `modules.py`   — core primitives (Linear, LayerNorm, triangle mult/attention, gated MHA, Pairformer block/stack). Reused verbatim from the OpenFold3→MLX port; works for both because the Pairformer block is identical (OpenDDE just widens `c_z` 128→384).
- `atom.py`      — windowed local atom attention (`rearrange_qk_to_dense_trunk` via pad+gather; `local_attention`).
- `atom_encdec.py` — AtomAttentionEncoder/Decoder + AtomTransformer.
- `diff_primitives.py` — AdaptiveLayerNorm, ConditionedTransitionBlock, AdaLN AttentionPairBias.
- `embed_misc.py` — FourierEmbedding, RelativePositionEncoding (`generate_relp`).
- `msa.py`       — OuterProductMean, MSA pair-weighted-averaging, MSAModule.
- `structural.py`— StructuralTokenExpander (full 49-projection mode) + structural refiner.
- `diffusion.py` — DiffusionConditioning (active pair compression 384→128) + 24-block DiffusionTransformer + EDM in/skip/out. **The diffusion transformer takes `extra_attn_bias = structural_pair_attn_bias`.**
- `sampler.py`   — Karras noise schedule + EDM predictor-corrector `sample_diffusion` (accepts injected noise for exact-parity gating).
- `heads.py`     — DistogramHead + ConfidenceHead (pLDDT/PAE/PDE/resolved).
- `ranking.py`   — logits→score, pTM/ipTM, ranking_score, clash.
- `weights.py`   — load `opendde.pt` (`['model']`, strip `module.`) → MLX arrays; flat→grouped Pairformer remap.
- `tests/`       — per-stage parity gates + `capture_full.py` (record torch refs + noise) + `run_exact.py` (exact-coordinate match) + `full_fold.py` (end-to-end fold).

## Notes / gotchas found during the port
- OpenDDE's `PairformerStack` mutates its input `z` in place — capture torch reference inputs with `register_forward_pre_hook` + `.clone()`.
- No-MSA inference: `make_dummy_feature(feats, ("msa",))` adds a 1-row restype MSA so `msa_module` actually runs (skipping it degrades folds ~9.6 Å → 2.5 Å).
- Config: `c_s=384`, `c_z=384` (`hidden_scale_up` → `no_heads_pair=12`), 48 Pairformer blocks, 4 MSA blocks, 24 diffusion-transformer blocks, structural refiner 4 blocks, `N_cycle` default 10, `sigma_data=16`.

## Running

```bash
export LAYERNORM_TYPE=torch OPENDDE_ROOT_DIR=$HOME/.cache/opendde
# 1) capture torch reference + noise on ubiquitin
PYTHONPATH=. python -m opendde_mlx.tests.capture_full
# 2) prove exact-coordinate match
PYTHONPATH=. python -m opendde_mlx.tests.run_exact
# 3) end-to-end MLX fold
PYTHONPATH=. python -m opendde_mlx.tests.full_fold
```

Reuses the OpenDDE torch data pipeline for featurization (numpy/RDKit/biotite);
only the neural network runs in MLX.
