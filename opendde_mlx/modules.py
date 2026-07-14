"""Core MLX primitives for the OpenFold3 port.

Each function mirrors the corresponding torch primitive in
``openfold3/core/model/primitives`` and ``.../layers`` exactly, so weights load
1:1 and outputs match to float32 precision.
"""

from __future__ import annotations

import math

import mlx.core as mx

# Toggle fused Metal kernels (mx.fast.layer_norm / scaled_dot_product_attention).
# These are hand-written kernels that mx.compile cannot synthesize.
USE_FAST = False


# ---------------------------------------------------------------------------
# Elementary primitives
# ---------------------------------------------------------------------------
def linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    """torch ``Linear``: weight is (out, in); computes ``x @ weight.T + bias``."""
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def layer_norm(
    x: mx.array,
    weight: mx.array | None,
    bias: mx.array | None,
    eps: float = 1e-5,
) -> mx.array:
    """LayerNorm over the last dim, matching torch ``F.layer_norm`` semantics."""
    if USE_FAST:
        return mx.fast.layer_norm(x, weight, bias, eps)
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    y = (x - mu) * mx.rsqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def sigmoid(x: mx.array) -> mx.array:
    return mx.sigmoid(x)


# ---------------------------------------------------------------------------
# Triangle multiplicative update (AF2 Alg. 11/12, AF3 Alg. 12/13)
# ---------------------------------------------------------------------------
def tri_mul(
    z: mx.array,
    p: dict,
    outgoing: bool,
    mask: mx.array | None = None,
    ln_eps: float = 1e-5,
) -> mx.array:
    """Triangle multiplicative update.

    Faithful to ``TriangleMultiplicativeUpdate.forward`` (non-inplace path):

        mask = mask[..., None]
        z    = layer_norm_in(z)
        a    = mask * sigmoid(linear_a_g(z)) * linear_a_p(z)
        b    = mask * sigmoid(linear_b_g(z)) * linear_b_p(z)
        x    = combine(a, b)                # outgoing:  ikc,jkc->ijc
                                            # incoming:  kic,kjc->ijc
        x    = linear_z(layer_norm_out(x))
        g    = sigmoid(linear_g(z))
        return x * g

    ``p`` maps the torch submodule names to MLX arrays:
        layer_norm_in.{weight,bias}, layer_norm_out.{weight,bias},
        linear_a_g.weight, linear_a_p.weight, linear_b_g.weight,
        linear_b_p.weight, linear_z.weight, linear_g.weight
    (these linears are bias-free in the checkpoint).
    """
    if mask is None:
        mask = mx.ones(z.shape[:-1], dtype=z.dtype)
    mask = mask[..., None]

    zn = layer_norm(z, p["layer_norm_in.weight"], p["layer_norm_in.bias"], ln_eps)

    a = mask * sigmoid(linear(zn, p["linear_a_g.weight"])) * linear(zn, p["linear_a_p.weight"])
    b = mask * sigmoid(linear(zn, p["linear_b_g.weight"])) * linear(zn, p["linear_b_p.weight"])

    if outgoing:
        x = mx.einsum("...ikc,...jkc->...ijc", a, b)
    else:
        x = mx.einsum("...kic,...kjc->...ijc", a, b)

    x = layer_norm(x, p["layer_norm_out.weight"], p["layer_norm_out.bias"], ln_eps)
    x = linear(x, p["linear_z.weight"])
    g = sigmoid(linear(zn, p["linear_g.weight"]))
    return x * g


# ---------------------------------------------------------------------------
# SwiGLU + transition (AF3 Alg. 11)
# ---------------------------------------------------------------------------
def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def swiglu(x: mx.array, wa: mx.array, wb: mx.array) -> mx.array:
    return silu(linear(x, wa)) * linear(x, wb)


def swiglu_transition(
    x: mx.array, p: dict, mask: mx.array | None = None, ln_eps: float = 1e-5
) -> mx.array:
    """SwiGLUTransition: layer_norm -> swiglu -> linear_out (-> * mask)."""
    xn = layer_norm(x, p["layer_norm.weight"], p["layer_norm.bias"], ln_eps)
    h = swiglu(xn, p["swiglu.linear_a.weight"], p["swiglu.linear_b.weight"])
    out = linear(h, p["linear_out.weight"])
    if mask is not None:
        out = out * mask[..., None]
    return out


# ---------------------------------------------------------------------------
# General gated multi-head attention with additive biases
# ---------------------------------------------------------------------------
def mha(
    q_x: mx.array,
    kv_x: mx.array,
    p: dict,
    no_heads: int,
    biases: tuple = (),
    gating: bool = True,
) -> mx.array:
    """Matches primitives.Attention: heads = c_hidden per head, q scaled by
    1/sqrt(c_hidden), additive biases before softmax, optional sigmoid gating."""
    H = no_heads

    def split(t):
        t = t.reshape(*t.shape[:-1], H, -1)
        return mx.swapaxes(t, -2, -3)  # [*, H, N, c_hidden]

    q = split(linear(q_x, p["linear_q.weight"], p.get("linear_q.bias")))
    k = split(linear(kv_x, p["linear_k.weight"], p.get("linear_k.bias")))
    v = split(linear(kv_x, p["linear_v.weight"], p.get("linear_v.bias")))

    c_hidden = q.shape[-1]
    scale = 1.0 / math.sqrt(c_hidden)

    if USE_FAST:
        mask = None
        if biases:
            mask = biases[0]
            for b in biases[1:]:
                mask = mask + b
        # mx.fast.scaled_dot_product_attention wants 4D [B, H, L, D].
        squeeze = q.ndim == 3
        if squeeze:
            q, k, v = q[None], k[None], v[None]
            if mask is not None:
                mask = mask[None]
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        if squeeze:
            o = o[0]
        o = mx.swapaxes(o, -2, -3)  # [*, Q, H, c_hidden]
    else:
        q = q * scale
        scores = mx.einsum("...qc,...kc->...qk", q, k)
        for b in biases:
            scores = scores + b
        scores = mx.softmax(scores, axis=-1)
        o = mx.einsum("...qk,...kc->...qc", scores, v)  # [*, H, Q, c_hidden]
        o = mx.swapaxes(o, -2, -3)  # [*, Q, H, c_hidden]

    if gating:
        g = mx.sigmoid(linear(q_x, p["linear_g.weight"], p.get("linear_g.bias")))
        g = g.reshape(*g.shape[:-1], H, -1)
        o = o * g

    o = o.reshape(*o.shape[:-2], -1)  # flatten heads
    return linear(o, p["linear_o.weight"], p.get("linear_o.bias"))


# ---------------------------------------------------------------------------
# Triangle attention (AF2 Alg. 13/14) — starting/ending node
# ---------------------------------------------------------------------------
def triangle_attention(
    x: mx.array,
    p: dict,
    no_heads: int,
    starting: bool = True,
    mask: mx.array | None = None,
    inf: float = 1e9,
    ln_eps: float = 1e-5,
) -> mx.array:
    """x: [*, I, J, C]. Ending node transposes I<->J first and back after.

    biases = [ mask_bias  = inf*(mask-1)[..., :, None, None, :],
               tri_bias   = permute(linear_z(x), (2,0,1)).unsqueeze(-4) ]
    then self-attention over the last-but-one axis via `mha`.
    """
    if not starting:
        x = mx.swapaxes(x, -2, -3)
        if mask is not None:
            mask = mx.swapaxes(mask, -1, -2)

    xn = layer_norm(x, p["layer_norm.weight"], p["layer_norm.bias"], ln_eps)

    biases = []
    if mask is not None:
        mask_bias = (inf * (mask - 1))[..., :, None, None, :]
        biases.append(mask_bias)

    tri = linear(xn, p["linear_z.weight"])  # [*, I, J, H]
    # permute_final_dims (2,0,1): (I, J, H) -> (H, I, J)
    tri = mx.moveaxis(tri, -1, -3)  # [*, H, I, J]
    tri = mx.expand_dims(tri, -4)  # unsqueeze(-4)
    biases.append(tri)

    mha_p = _sub(p, "mha")
    out = mha(xn, xn, mha_p, no_heads, biases=tuple(biases))

    if not starting:
        out = mx.swapaxes(out, -2, -3)
    return out


# ---------------------------------------------------------------------------
# Pairformer block (AF3 Alg. 17) and stack
# ---------------------------------------------------------------------------
def _sub(p: dict, prefix: str) -> dict:
    prefix = prefix + "."
    return {k[len(prefix):]: v for k, v in p.items() if k.startswith(prefix)}


def attention_pair_bias(
    a: mx.array,
    z: mx.array,
    p: dict,
    no_heads: int,
    mask: mx.array | None = None,
    inf: float = 1e9,
    ln_eps: float = 1e-5,
) -> mx.array:
    """AttentionPairBias (non-AdaLN): gated MHA over the single rep, biased by
    the pair rep. Returns the update (added to `a` by the caller)."""
    an = layer_norm(a, p["layer_norm_a.weight"], p["layer_norm_a.bias"], ln_eps)

    biases = []
    if mask is not None:
        biases.append((inf * (mask - 1))[..., None, None, :])

    zb = layer_norm(z, p["layer_norm_z.weight"], p["layer_norm_z.bias"], ln_eps)
    zb = linear(zb, p["linear_z.weight"])       # [*, N, N, H]
    zb = mx.moveaxis(zb, -1, -3)                 # permute_final_dims (2,0,1) -> [*, H, N, N]
    biases.append(zb)

    return mha(an, an, _sub(p, "mha"), no_heads, biases=tuple(biases), gating=True)


def pair_block(
    z: mx.array,
    p: dict,
    pair_mask: mx.array,
    no_heads_pair: int,
    inf: float = 1e9,
    ln_eps: float = 1e-5,
) -> mx.array:
    """PairBlock (pair_stack): tri-mul out/in, tri-att start/end, pair transition.
    Both tri-att layers are starting-node; the ending update transposes z externally."""
    z = z + tri_mul(z, _sub(p, "tri_mul_out"), outgoing=True, mask=pair_mask, ln_eps=ln_eps)
    z = z + tri_mul(z, _sub(p, "tri_mul_in"), outgoing=False, mask=pair_mask, ln_eps=ln_eps)

    z = z + triangle_attention(
        z, _sub(p, "tri_att_start"), no_heads_pair, starting=True,
        mask=pair_mask, inf=inf, ln_eps=ln_eps,
    )

    zt = mx.swapaxes(z, -2, -3)
    zt = zt + triangle_attention(
        zt, _sub(p, "tri_att_end"), no_heads_pair, starting=True,
        mask=mx.swapaxes(pair_mask, -1, -2), inf=inf, ln_eps=ln_eps,
    )
    z = mx.swapaxes(zt, -2, -3)

    z = z + swiglu_transition(z, _sub(p, "pair_transition"), mask=pair_mask, ln_eps=ln_eps)
    return z


def pairformer_block(
    s: mx.array,
    z: mx.array,
    p: dict,
    single_mask: mx.array,
    pair_mask: mx.array,
    no_heads_pair_bias: int = 16,
    no_heads_pair: int = 4,
    inf: float = 1e9,
    ln_eps: float = 1e-5,
) -> tuple[mx.array, mx.array]:
    z = pair_block(z, _sub(p, "pair_stack"), pair_mask, no_heads_pair, inf, ln_eps)
    s = s + attention_pair_bias(
        s, z, _sub(p, "attn_pair_bias"), no_heads_pair_bias,
        mask=single_mask, inf=inf, ln_eps=ln_eps,
    )
    s = s + swiglu_transition(s, _sub(p, "single_transition"), mask=single_mask, ln_eps=ln_eps)
    return s, z


def pairformer_stack(
    s: mx.array,
    z: mx.array,
    p: dict,
    single_mask: mx.array,
    pair_mask: mx.array,
    n_blocks: int,
    **kw,
) -> tuple[mx.array, mx.array]:
    for i in range(n_blocks):
        s, z = pairformer_block(s, z, _sub(p, f"blocks.{i}"), single_mask, pair_mask, **kw)
    return s, z
