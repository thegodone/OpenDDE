"""Diffusion-transformer building blocks (MLX port).

Ports three torch modules used by the OpenDDE diffusion transformer:

- ``AdaptiveLayerNorm``            (primitives.py, Alg. 26)
- ``ConditionedTransitionBlock``  (transformer.py, Alg. 25)
- ``AttentionPairBias`` (has_s=True AdaLN path, non-local token attention;
                          transformer.py, Alg. 24)

All three take a conditioning single embedding ``s`` and mix it in through an
AdaptiveLayerNorm + adaLN-Zero output gate.  Weight-key names mirror the torch
submodule names 1:1 so a raw checkpoint subtree loads directly.
"""

from __future__ import annotations

import mlx.core as mx

from opendde_mlx.modules import _sub, layer_norm, linear, mha, sigmoid, silu


# ---------------------------------------------------------------------------
# AdaptiveLayerNorm (Algorithm 26)
# ---------------------------------------------------------------------------
def adaptive_layer_norm(
    a: mx.array,
    s: mx.array,
    p: dict,
    ln_eps: float = 1e-5,
) -> mx.array:
    """AdaLN:  a = sigmoid(linear_s(LN(s))) * LN_noaffine(a) + linear_nobias_s(LN(s)).

    torch keys in ``p``:
        layernorm_s.weight            (LayerNorm over s, create_offset=False -> no bias)
        linear_s.{weight,bias}        (zero-init in torch; has bias)
        linear_nobias_s.weight        (zero-init in torch; no bias)

    ``layernorm_a`` in torch has create_scale=False, create_offset=False, so it
    carries no parameters — a bare LayerNorm.
    """
    a = layer_norm(a, None, None, ln_eps)
    sn = layer_norm(s, p["layernorm_s.weight"], None, ln_eps)
    a = sigmoid(linear(sn, p["linear_s.weight"], p["linear_s.bias"])) * a \
        + linear(sn, p["linear_nobias_s.weight"])
    return a


# ---------------------------------------------------------------------------
# ConditionedTransitionBlock (Algorithm 25)
# ---------------------------------------------------------------------------
def conditioned_transition_block(
    a: mx.array,
    s: mx.array,
    p: dict,
    ln_eps: float = 1e-5,
) -> mx.array:
    """AdaLN -> SwiGLU-style transition -> adaLN-Zero output gate.

        a = adaln(a, s)
        b = silu(linear_a1(a)) * linear_a2(a)
        a = sigmoid(linear_s(s)) * linear_b(b)

    torch keys in ``p``:
        adaln.*                       (see adaptive_layer_norm)
        linear_nobias_a1.weight
        linear_nobias_a2.weight
        linear_nobias_b.weight
        linear_s.{weight,bias}        (BiasInitLinear, biasinit=-2.0)
    """
    a = adaptive_layer_norm(a, s, _sub(p, "adaln"), ln_eps)
    b = silu(linear(a, p["linear_nobias_a1.weight"])) * linear(a, p["linear_nobias_a2.weight"])
    a = sigmoid(linear(s, p["linear_s.weight"], p["linear_s.bias"])) \
        * linear(b, p["linear_nobias_b.weight"])
    return a


# ---------------------------------------------------------------------------
# AttentionPairBias (Algorithm 24, has_s=True AdaLN path, standard/non-local)
# ---------------------------------------------------------------------------
def attention_pair_bias_adaln(
    a: mx.array,
    s: mx.array,
    z: mx.array,
    p: dict,
    no_heads: int = 16,
    ln_eps: float = 1e-5,
) -> mx.array:
    """Diffusion-transformer AttentionPairBias with conditioning ``s``.

    Non-local (full) token attention, cross_attention_mode=False:

        a  = adaln(a, s)                       # AdaptiveLayerNorm on the single rep
        kv = a
        bias = linear_nobias_z(LN(z))          # [..., N, N, H]
        bias = permute -> [..., H, N, N]
        a  = Attention(q=a, kv=kv, attn_bias=bias)   # gated MHA, linear_q has bias
        a  = sigmoid(linear_a_last(s)) * a     # adaLN-Zero output gate

    torch keys in ``p``:
        layernorm_a.*                 (AdaptiveLayerNorm subtree)
        layernorm_z.weight            (LayerNorm on z, create_offset=False -> no bias)
        linear_nobias_z.weight        (c_z -> n_heads)
        attention.*                   (linear_q{.weight,.bias}, linear_k/v/o/g.weight)
        linear_a_last.{weight,bias}   (BiasInitLinear, biasinit=-2.0)

    Note: the torch Attention here uses c_hidden = c_a // n_heads and scales q by
    1/sqrt(c_hidden); ``mha`` reproduces this exactly (scale from q's last dim).
    """
    a = adaptive_layer_norm(a, s, _sub(p, "layernorm_a"), ln_eps)
    kv = a

    zb = layer_norm(z, p["layernorm_z.weight"], None, ln_eps)
    zb = linear(zb, p["linear_nobias_z.weight"])   # [..., N, N, H]
    zb = mx.moveaxis(zb, -1, -3)                    # [..., H, N, N]

    a = mha(a, kv, _sub(p, "attention"), no_heads, biases=(zb,), gating=True)

    a = sigmoid(linear(s, p["linear_a_last.weight"], p["linear_a_last.bias"])) * a
    return a
