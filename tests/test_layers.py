"""Block-level invariants: RoPE, attention, adaLN conditioning, equivariance."""

from __future__ import annotations

import mlx.core as mx
import pytest

from loopmdm.config import ModelConfig
from loopmdm.layers import (
    adaln_modulation,
    apply_rope,
    attention,
    block_forward,
    init_layer_params,
    init_time_params,
    mlp,
    rms_norm,
    rope_tables,
    sinusoidal_embedding,
    time_mlp,
)


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=16, seq_len=8, d_model=16, n_heads=4, n_layers=3,
        head_layers=1, mid_layers=1, time_dim=16,
    )


def test_rms_norm_hand_computed():
    x = mx.array([[3.0, 4.0]])
    w = mx.ones((2,))
    out = rms_norm(x, w)
    scale = 1.0 / (12.5 + 1e-5) ** 0.5
    assert out.tolist()[0] == pytest.approx([3.0 * scale, 4.0 * scale], rel=1e-5)
    # the weight is applied after normalisation
    assert rms_norm(x, mx.array([2.0, 3.0])).tolist()[0] == pytest.approx(
        [6.0 * scale, 12.0 * scale], rel=1e-5
    )


def test_rms_norm_scales_invariant():
    x = mx.random.normal((2, 3, 8), key=mx.random.key(0))
    w = mx.random.normal((8,), key=mx.random.key(1))
    a = rms_norm(x, w)
    b = rms_norm(x * 7.0, w)
    assert mx.allclose(a, b, atol=1e-4).item()


def test_sinusoidal_embedding_is_bounded_and_time_dependent():
    t = mx.array([0.0, 0.5, 1.0])
    emb = sinusoidal_embedding(t, 8)
    assert emb.shape == (3, 8)
    assert float(mx.max(mx.abs(emb))) <= 1.0 + 1e-6
    assert not mx.allclose(emb[0], emb[1]).item()
    with pytest.raises(ValueError):
        sinusoidal_embedding(t, 7)


def test_time_mlp_shape(cfg):
    params = init_time_params(mx.random.key(0), cfg)
    out = time_mlp(params, mx.array([0.25, 0.75]), cfg)
    assert out.shape == (2, cfg.time_dim)
    assert out.dtype == mx.float32


def test_rope_is_identity_at_position_zero_and_preserves_norms():
    cos, sin = rope_tables(6, 8)
    x = mx.random.normal((2, 2, 6, 8), key=mx.random.key(0))
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
    assert mx.allclose(out[:, :, 0], x[:, :, 0], atol=1e-6).item()
    before = mx.sum(mx.square(x), axis=-1)
    after = mx.sum(mx.square(out), axis=-1)
    assert mx.allclose(before, after, atol=1e-4).item()


def test_rope_dot_products_depend_only_on_relative_position():
    t, dh = 7, 8
    cos, sin = rope_tables(t, dh)
    # the same vector at every position: then only the *difference* of the two
    # rotated positions may matter
    qvec = mx.random.normal((1, 1, 1, dh), key=mx.random.key(2))
    kvec = mx.random.normal((1, 1, 1, dh), key=mx.random.key(3))
    q0 = mx.broadcast_to(qvec, (1, 1, t, dh))
    k0 = mx.broadcast_to(kvec, (1, 1, t, dh))
    q = apply_rope(q0, cos, sin)
    k = apply_rope(k0, cos, sin)
    dots = (q @ k.swapaxes(-1, -2))[0, 0]
    for i in range(t):
        for j in range(t):
            offset = i - j
            ref_i, ref_j = (offset, 0) if offset >= 0 else (0, -offset)
            assert float(dots[i, j]) == pytest.approx(float(dots[ref_i, ref_j]), abs=1e-4)


def _project(x, w, n_heads):
    """Project to heads; no RoPE here (values are never rotated)."""
    b, t, d = x.shape
    dh = d // n_heads
    return (x @ w.T).reshape(b, t, n_heads, dh).transpose(0, 2, 1, 3)


def test_attention_matches_the_fused_kernel(cfg):
    x = mx.random.normal((2, cfg.seq_len, cfg.d_model), key=mx.random.key(4))
    layer = init_layer_params(mx.random.key(5), cfg)
    cos, sin = rope_tables(cfg.seq_len, cfg.d_head)
    out, probs = attention(layer["attn"], x, cos, sin, cfg.n_heads, capture=True)

    q = apply_rope(_project(x, layer["attn"]["wq"], cfg.n_heads), cos, sin)
    k = apply_rope(_project(x, layer["attn"]["wk"], cfg.n_heads), cos, sin)
    v = _project(x, layer["attn"]["wv"], cfg.n_heads)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=cfg.d_head ** -0.5)
    ref = ref.transpose(0, 2, 1, 3).reshape(2, cfg.seq_len, cfg.d_model) @ layer["attn"]["wo"].T
    assert mx.allclose(out, ref, atol=1e-5).item()
    assert probs.shape == (2, cfg.n_heads, cfg.seq_len, cfg.seq_len)
    assert mx.allclose(mx.sum(probs, axis=-1), mx.ones((2, cfg.n_heads, cfg.seq_len)), atol=1e-5).item()


def test_attention_is_bidirectional(cfg):
    """A masked diffusion denoiser attends in both directions: no position is masked out."""
    x = mx.random.normal((1, cfg.seq_len, cfg.d_model), key=mx.random.key(6))
    layer = init_layer_params(mx.random.key(7), cfg)
    cos, sin = rope_tables(cfg.seq_len, cfg.d_head)
    _, probs = attention(layer["attn"], x, cos, sin, cfg.n_heads, capture=True)
    assert float(mx.min(probs)) > 0.0
    # changing a *later* token must change an earlier position's output
    out_a, _ = attention(layer["attn"], x, cos, sin, cfg.n_heads)
    x2 = mx.concatenate([x[:, :-1], x[:, -1:] * 0.0 + 5.0], axis=1)
    out_b, _ = attention(layer["attn"], x2, cos, sin, cfg.n_heads)
    assert not mx.allclose(out_a[:, 0], out_b[:, 0], atol=1e-6).item()


def test_block_with_zero_adaln_is_a_plain_prenorm_block(cfg):
    """Zeroing the adaLN modulation must leave exactly a plain pre-norm block."""
    layer = init_layer_params(mx.random.key(8), cfg)
    layer["ada"]["w"] = mx.zeros_like(layer["ada"]["w"])
    layer["ada"]["b"] = mx.zeros_like(layer["ada"]["b"])
    x = mx.random.normal((1, cfg.seq_len, cfg.d_model), key=mx.random.key(9))
    t_emb = time_mlp(init_time_params(mx.random.key(10), cfg), mx.array([0.4]), cfg)
    cos, sin = rope_tables(cfg.seq_len, cfg.d_head)
    out, _ = block_forward(layer, x, t_emb, cos, sin, cfg)

    shift_a, scale_a, shift_m, scale_m = adaln_modulation(layer, t_emb)
    assert float(mx.max(mx.abs(shift_a))) == 0.0
    assert float(mx.max(mx.abs(scale_m))) == 0.0
    h = rms_norm(x, layer["norm1"])
    h = x + attention(layer["attn"], h, cos, sin, cfg.n_heads)[0]
    h = h + mlp(layer["mlp"], rms_norm(h, layer["norm2"]))
    assert mx.allclose(out, h, atol=1e-5).item()


def test_block_output_shape_and_dtype_and_permutation_equivariance(cfg):
    layer = init_layer_params(mx.random.key(11), cfg)
    x = mx.random.normal((2, cfg.seq_len, cfg.d_model), key=mx.random.key(12))
    t_emb = time_mlp(init_time_params(mx.random.key(13), cfg), mx.array([0.2, 0.9]), cfg)
    cos, sin = rope_tables(cfg.seq_len, cfg.d_head)
    out, _ = block_forward(layer, x, t_emb, cos, sin, cfg)
    assert out.shape == x.shape and out.dtype == mx.float32

    # no positional information enters the block outside attention's RoPE, so the
    # block must commute with token order (RoPE breaks this only across positions)
    perm = mx.array([3, 0, 5, 1, 7, 2, 6, 4])
    permuted, _ = block_forward(
        layer, mx.take(x, perm, axis=1), t_emb, cos, sin, cfg
    )
    assert mx.allclose(mx.take(out, perm, axis=1), permuted, atol=1e-4).item()


def test_capture_returns_none_by_default(cfg):
    layer = init_layer_params(mx.random.key(14), cfg)
    x = mx.random.normal((1, cfg.seq_len, cfg.d_model), key=mx.random.key(15))
    cos, sin = rope_tables(cfg.seq_len, cfg.d_head)
    _, probs = attention(layer["attn"], x, cos, sin, cfg.n_heads)
    assert probs is None
