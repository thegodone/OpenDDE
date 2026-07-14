"""MLX port of two OpenDDE embedding utilities (opendde/model/modules/embedders.py):

- ``FourierEmbedding`` (Algorithm 22) — cos(2*pi*(t*w + b)); the ``w``/``b`` buffers
  are seed-42 random *and stored in the checkpoint*, so they MUST be loaded, not
  regenerated (torch/mlx RNG differ).
- ``RelativePositionEncoding.generate_relp`` (Algorithm 3) — builds the one-hot /
  clip relative-position feature block [..., N, N, 139], plus the trailing
  linear_no_bias projection [..., N, N, c_z].

Reuses ``opendde_mlx.modules.linear``. p-dicts keep the torch key names verbatim.
"""
from __future__ import annotations

import math

import mlx.core as mx

from opendde_mlx.modules import linear


# --------------------------------------------------------------------------- #
# FourierEmbedding (Algorithm 22)
# --------------------------------------------------------------------------- #
def fourier_embedding(t_hat_noise_level: mx.array, p: dict) -> mx.array:
    """cos(2*pi*(t[..., None] * w + b)).

    Args:
        t_hat_noise_level: [..., N_sample]
        p: dict with keys ``w`` and ``b`` (both shape (c,)).
    Returns:
        [..., N_sample, c]
    """
    w = p["w"]
    b = p["b"]
    return mx.cos(2.0 * math.pi * (t_hat_noise_level[..., None] * w + b))


# --------------------------------------------------------------------------- #
# RelativePositionEncoding (Algorithm 3)
# --------------------------------------------------------------------------- #
def _one_hot(idx: mx.array, num_classes: int) -> mx.array:
    """Integer index tensor -> one-hot along a new trailing axis (float32),
    matching torch.nn.functional.one_hot semantics."""
    idx = idx.astype(mx.int32)
    classes = mx.arange(num_classes, dtype=mx.int32)
    return (idx[..., None] == classes).astype(mx.float32)


def generate_relp(
    input_feature_dict: dict,
    r_max: int = 32,
    s_max: int = 2,
) -> mx.array:
    """Port of ``RelativePositionEncoding.generate_relp`` (non-lazy path).

    Consumes ``asym_id``, ``residue_index``, ``entity_id``, ``token_index``,
    ``sym_id`` (each [..., N_token]) and returns relp [..., N_token, N_token, 139]
    where 139 = 4*r_max + 2*s_max + 7 (= 66 + 66 + 1 + 6 for r_max=32, s_max=2).
    """
    asym_id = input_feature_dict["asym_id"]
    residue_index = input_feature_dict["residue_index"]
    entity_id = input_feature_dict["entity_id"]
    token_index = input_feature_dict["token_index"]
    sym_id = input_feature_dict["sym_id"]

    def _pairwise_eq(x):
        return (x[..., :, None] == x[..., None, :]).astype(mx.int32)

    b_same_chain = _pairwise_eq(asym_id)      # [..., N, N]
    b_same_residue = _pairwise_eq(residue_index)
    b_same_entity = _pairwise_eq(entity_id)

    d_residue = mx.clip(
        residue_index[..., :, None] - residue_index[..., None, :] + r_max,
        0, 2 * r_max,
    ) * b_same_chain + (1 - b_same_chain) * (2 * r_max + 1)
    a_rel_pos = _one_hot(d_residue, 2 * (r_max + 1))            # 2*(r_max+1)

    d_token = mx.clip(
        token_index[..., :, None] - token_index[..., None, :] + r_max,
        0, 2 * r_max,
    ) * b_same_chain * b_same_residue + (
        1 - b_same_chain * b_same_residue
    ) * (2 * r_max + 1)
    a_rel_token = _one_hot(d_token, 2 * (r_max + 1))            # 2*(r_max+1)

    d_chain = mx.clip(
        sym_id[..., :, None] - sym_id[..., None, :] + s_max,
        0, 2 * s_max,
    ) * b_same_entity + (1 - b_same_entity) * (2 * s_max + 1)
    a_rel_chain = _one_hot(d_chain, 2 * (s_max + 1))            # 2*(s_max+1)

    relp = mx.concatenate(
        [a_rel_pos, a_rel_token, b_same_entity[..., None].astype(mx.float32), a_rel_chain],
        axis=-1,
    )
    return relp


def relative_position_encoding(relp_feature: mx.array, p: dict) -> mx.array:
    """Trailing projection: ``linear_no_bias`` [.., N, N, 139] -> [.., N, N, c_z]."""
    return linear(relp_feature, p["linear_no_bias.weight"])
