"""Load the OpenDDE torch checkpoint into MLX and remap its flat PairformerBlock
layout onto the grouped param-dict convention used by opendde_mlx.modules
(reused verbatim from the openfold-3-mlx port)."""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx


def load_state_dict(path: str | Path) -> dict:
    """opendde.pt -> flat dict[str, mx.array] (float32). Container key is ['model'];
    keys carry a leading 'module.' (DDP) which is stripped."""
    import torch

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    out = {}
    for k, v in sd.items():
        if not hasattr(v, "detach"):
            continue
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = mx.array(v.detach().to(torch.float32).numpy())
    return out


def subtree(sd: dict, prefix: str) -> dict:
    if not prefix.endswith("."):
        prefix += "."
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


# --- OpenDDE flat PairformerBlock  ->  opendde_mlx.modules grouped p-dict --------
# Applied to the keys of ONE block subtree (prefix already stripped).
def remap_pairformer_block(bp: dict) -> dict:
    """Rename a single OpenDDE PairformerBlock's flat keys to what
    modules.pairformer_block expects (pair_stack.* grouping + helper names)."""
    out = {}
    for k, v in bp.items():
        nk = k
        # triangle multiplication: names already match, just group under pair_stack.
        if k.startswith("tri_mul_out.") or k.startswith("tri_mul_in."):
            nk = "pair_stack." + k
        # triangle attention: bias proj `linear` -> `linear_z`; group under pair_stack.
        elif k.startswith("tri_att_start.") or k.startswith("tri_att_end."):
            nk = "pair_stack." + k.replace(".linear.weight", ".linear_z.weight")
        # pair transition (SwiGLU): layernorm1->layer_norm, a/b->swiglu.*, _ ->linear_out
        elif k.startswith("pair_transition."):
            nk = "pair_stack." + _remap_transition(k, "pair_transition.")
        elif k.startswith("single_transition."):
            nk = _remap_transition(k, "single_transition.")
        # attention with pair bias: attention_pair_bias -> attn_pair_bias, submodule renames
        elif k.startswith("attention_pair_bias."):
            nk = ("attn_pair_bias." + k[len("attention_pair_bias."):]
                  .replace("layernorm_a.", "layer_norm_a.")
                  .replace("layernorm_z.", "layer_norm_z.")
                  .replace("linear_nobias_z.", "linear_z.")
                  .replace("attention.", "mha."))
        out[nk] = v
    return out


def _remap_transition(k: str, prefix: str) -> str:
    tail = k[len(prefix):]
    tail = (tail.replace("layernorm1.", "layer_norm.")
                .replace("linear_no_bias_a.", "swiglu.linear_a.")
                .replace("linear_no_bias_b.", "swiglu.linear_b.")
                .replace("linear_no_bias.", "linear_out."))
    return prefix + tail


def remap_pairformer_stack(sd: dict, stack_prefix: str) -> dict:
    """Remap all blocks of a pairformer stack -> grouped 'blocks.{i}.*' p-dict."""
    st = subtree(sd, stack_prefix)
    n_blocks = 1 + max(int(m.group(1)) for k in st
                       if (m := re.match(r"blocks\.(\d+)\.", k)))
    out = {}
    for i in range(n_blocks):
        bp = subtree(st, f"blocks.{i}")
        for nk, v in remap_pairformer_block(bp).items():
            out[f"blocks.{i}.{nk}"] = v
    return out, n_blocks
