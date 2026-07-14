"""Transformer blocks for the denoising network.

One block is pre-norm RMSNorm -> multi-head self-attention -> RMSNorm -> MLP,
with a diffusion-timestep conditioning vector modulating both sublayers in the
style of a DiT block (scale/shift, no gate).  Attention is *bidirectional*: a
masked diffusion denoiser sees the whole partially-masked sequence, so there is
no causal mask.  Positions are encoded with rotary embeddings applied to the
queries and keys only, so a block is permutation-equivariant.

Every function here is written over an explicit parameter pytree rather than an
``nn.Module``.  That is what makes weight sharing across loop iterations exact:
the same arrays are read ``R`` times by the same traced function, so MLX's
reverse-mode AD accumulates the gradient contributions of all iterations.
"""

from __future__ import annotations

import math

import mlx.core as mx

from .config import ModelConfig

EPS = 1e-5


def silu(x: mx.array) -> mx.array:
    """SiLU / swish activation."""
    return x * mx.sigmoid(x)


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def rms_norm(x: mx.array, weight: mx.array, eps: float = EPS) -> mx.array:
    """RMSNorm: ``x / sqrt(mean(x^2) + eps) * weight`` (no mean subtraction)."""
    ms = mx.mean(mx.square(x), axis=-1, keepdims=True)
    return x * mx.rsqrt(ms + eps) * weight


def sinusoidal_embedding(t: mx.array, dim: int, base: float = 10000.0) -> mx.array:
    """Standard sinusoidal features of a scalar ``t`` in [0, 1] (shape ``(B, dim)``)."""
    if dim % 2 != 0:
        raise ValueError("sinusoidal_embedding needs an even dim")
    half = dim // 2
    freqs = mx.exp(-math.log(base) * mx.arange(half, dtype=mx.float32) / half)
    args = t.reshape(-1, 1) * freqs.reshape(1, -1)
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def rope_tables(seq_len: int, d_head: int, base: float = 10000.0) -> tuple[mx.array, mx.array]:
    """Cosine/sine tables of shape ``(seq_len, d_head // 2)``."""
    half = d_head // 2
    inv = mx.exp(-math.log(base) * mx.arange(half, dtype=mx.float32) / half)
    pos = mx.arange(seq_len, dtype=mx.float32)
    angles = pos.reshape(-1, 1) * inv.reshape(1, -1)
    return mx.cos(angles), mx.sin(angles)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate adjacent feature pairs by ``pos * inv_freq``.

    ``x`` has shape ``(B, H, T, D)`` and ``cos``/``sin`` broadcast over the
    head dimension.
    """
    d = x.shape[-1]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    c = cos.reshape(1, 1, *cos.shape)
    s = sin.reshape(1, 1, *sin.shape)
    out = mx.stack([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)
    return out.reshape(*x.shape[:-1], d)


def time_mlp(params: dict, t: mx.array, cfg: ModelConfig) -> mx.array:
    """Map the scalar diffusion time to a conditioning vector ``(B, time_dim)``."""
    emb = sinusoidal_embedding(t, cfg.time_dim)
    h = emb @ params["w1"].T + params["b1"]
    h = silu(h)
    return h @ params["w2"].T + params["b2"]


def attention(
    params: dict,
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    n_heads: int,
    capture: bool = False,
) -> tuple[mx.array, mx.array | None]:
    """Bidirectional multi-head self-attention with RoPE.

    Returns ``(output, attention_probs)``; the probabilities are only returned
    when ``capture`` is set (they are what the mask-to-mask analysis measures).
    """
    b, t, d = x.shape
    dh = d // n_heads
    q = (x @ params["wq"].T).reshape(b, t, n_heads, dh).transpose(0, 2, 1, 3)
    k = (x @ params["wk"].T).reshape(b, t, n_heads, dh).transpose(0, 2, 1, 3)
    v = (x @ params["wv"].T).reshape(b, t, n_heads, dh).transpose(0, 2, 1, 3)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    scores = (q @ k.swapaxes(-1, -2)) * (dh ** -0.5)
    probs = mx.softmax(scores, axis=-1)
    out = probs @ v
    out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
    return out @ params["wo"].T, (probs if capture else None)


def mlp(params: dict, x: mx.array) -> mx.array:
    h = x @ params["w1"].T + params["b1"]
    h = silu(h)
    return h @ params["w2"].T + params["b2"]


def adaln_modulation(params: dict, t_emb: mx.array) -> tuple[mx.array, ...]:
    """Per-block (shift, scale) pairs for attention and MLP from the time vector."""
    mod = silu(t_emb) @ params["ada"]["w"].T + params["ada"]["b"]
    shift_a, scale_a, shift_m, scale_m = mx.split(mod, 4, axis=-1)
    return shift_a, scale_a, shift_m, scale_m


def block_forward(
    layer: dict,
    x: mx.array,
    t_emb: mx.array,
    cos: mx.array,
    sin: mx.array,
    cfg: ModelConfig,
    capture: bool = False,
) -> tuple[mx.array, mx.array | None]:
    """One transformer block; returns ``(hidden, attention_probs_or_None)``."""
    shift_a, scale_a, shift_m, scale_m = adaln_modulation(layer, t_emb)
    h = rms_norm(x, layer["norm1"])
    h = h * (1.0 + scale_a[:, None, :]) + shift_a[:, None, :]
    attn_out, probs = attention(layer["attn"], h, cos, sin, cfg.n_heads, capture)
    x = x + attn_out
    h = rms_norm(x, layer["norm2"])
    h = h * (1.0 + scale_m[:, None, :]) + shift_m[:, None, :]
    x = x + mlp(layer["mlp"], h)
    return x, probs


# --------------------------------------------------------------------------- #
# initialisation
# --------------------------------------------------------------------------- #
def init_layer_params(key: mx.array, cfg: ModelConfig) -> dict:
    """Initialise one block.

    Weights are ``N(0, 0.02)`` and biases are zero.  DiT zero-initialises the
    adaLN projection; here it is deliberately *not* zero-initialised, because a
    zero modulation would make the denoiser independent of the diffusion time
    ``t`` for the first optimisation steps, and these toy runs are short.
    """
    keys = mx.random.split(key, 7)
    d, h = cfg.d_model, cfg.mlp_hidden

    def lin(out_dim: int, in_dim: int, k: mx.array) -> mx.array:
        return mx.random.normal((out_dim, in_dim), key=k) * 0.02

    return {
        "norm1": mx.ones((d,)),
        "norm2": mx.ones((d,)),
        # attention projections carry no bias, so a zero-modulated block is exactly
        # a plain pre-norm attention block
        "attn": {name: lin(d, d, keys[i]) for i, name in enumerate(("wq", "wk", "wv", "wo"))},
        "mlp": {
            "w1": lin(h, d, keys[4]), "b1": mx.zeros((h,)),
            "w2": lin(d, h, keys[5]), "b2": mx.zeros((d,)),
        },
        "ada": {
            "w": mx.random.normal((4 * d, cfg.time_dim), key=keys[6]) * 0.02,
            "b": mx.zeros((4 * d,)),
        },
    }


def init_time_params(key: mx.array, cfg: ModelConfig) -> dict:
    keys = mx.random.split(key, 2)
    dim = cfg.time_dim
    return {
        "w1": mx.random.normal((dim, dim), key=keys[0]) * 0.02,
        "b1": mx.zeros((dim,)),
        "w2": mx.random.normal((dim, dim), key=keys[1]) * 0.02,
        "b2": mx.zeros((dim,)),
    }
