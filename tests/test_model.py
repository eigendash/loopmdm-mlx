"""Invariants of the looped model: exactness at R=1, tying, cost, gradients."""

from __future__ import annotations

import mlx.core as mx
import mlx.utils as mu
import pytest

from loopmdm.config import ModelConfig
from loopmdm.layers import block_forward, rms_norm, rope_tables, time_mlp
from loopmdm.model import (
    count_params,
    forward,
    forward_adaptive,
    init_params,
    layer_applications,
    num_parameters,
    relative_hidden_change,
    unroll_params,
    unrolled_config,
)


@pytest.fixture
def cfg():
    # head + mid + tail = 1 + 2 + 1 = 4 layers
    return ModelConfig(
        vocab_size=12, seq_len=6, d_model=16, n_heads=4, n_layers=4,
        head_layers=1, mid_layers=2, time_dim=16,
    )


@pytest.fixture
def batch(cfg):
    tokens = mx.random.randint(0, cfg.padded_vocab, shape=(3, cfg.seq_len), key=mx.random.key(0))
    t = mx.array([0.1, 0.5, 0.9])
    return tokens, t


def test_loop_count_one_is_bit_identical_to_the_unlooped_model(cfg, batch):
    """R = 1 is the clean limit: exactly the same graph as a non-looped model."""
    tokens, t = batch
    params = init_params(cfg, mx.random.key(1))
    looped = forward(params, tokens, t, cfg, loop_count=1)
    plain = forward(unroll_params(params, 1), tokens, t, unrolled_config(cfg, 1), loop_count=1)
    assert mx.array_equal(looped.logits, plain.logits).item()
    assert looped.logits.tolist() == plain.logits.tolist()


@pytest.mark.parametrize("loop_count", [1, 2, 3, 4])
def test_looped_model_equals_a_deeper_model_with_tied_weights(cfg, batch, loop_count):
    """Applying the mid block R times == an R*n_m layer deep model sharing weights."""
    tokens, t = batch
    params = init_params(cfg, mx.random.key(2))
    looped = forward(params, tokens, t, cfg, loop_count=loop_count)
    deep_cfg = unrolled_config(cfg, loop_count)
    assert deep_cfg.n_layers == cfg.n_layers + (loop_count - 1) * cfg.mid_layers
    deep = forward(unroll_params(params, loop_count), tokens, t, deep_cfg, loop_count=1)
    assert mx.allclose(looped.logits, deep.logits, atol=1e-6).item()
    assert deep.logits.shape == looped.logits.shape


@pytest.mark.parametrize("loop_count", [1, 2, 3, 8])
def test_looping_adds_zero_parameters(cfg, loop_count):
    params = init_params(cfg, mx.random.key(3))
    assert count_params(params) == num_parameters(cfg)
    assert count_params(unroll_params(params, loop_count)) == count_params(params)
    # the unrolled tree really does repeat the same arrays (tied, not copied)
    unrolled = unroll_params(params, loop_count)
    assert len(unrolled["mid"]) == loop_count * cfg.mid_layers
    assert unrolled["mid"][0] is params["mid"][0]
    for r in range(loop_count):
        assert unrolled["mid"][r * cfg.mid_layers] is params["mid"][0]


@pytest.mark.parametrize("loop_count", [1, 2, 5])
def test_layer_applications_match_the_predicted_count(cfg, batch, loop_count):
    tokens, t = batch
    params = init_params(cfg, mx.random.key(4))
    out = forward(params, tokens, t, cfg, loop_count=loop_count, capture=True)
    predicted = cfg.head_layers + loop_count * cfg.mid_layers + cfg.tail_layers
    assert predicted == layer_applications(cfg, loop_count)
    assert out.n_blocks == predicted
    assert len(out.attentions) == predicted
    # cost is linear in R with slope n_m
    assert layer_applications(cfg, loop_count) - layer_applications(cfg, 1) == (
        loop_count - 1
    ) * cfg.mid_layers


def _mid_leaf(tree):
    return [tree["mid"][i] for i in range(len(tree["mid"]))]


def _deep_copy(tree):
    """A structurally identical tree of *new* arrays (so MLX cannot tie them)."""
    return mu.tree_map(lambda a: mx.array(a), tree)


def _grad_mass(tree) -> float:
    return sum(float(mx.sum(mx.abs(a)).item()) for _, a in mu.tree_flatten(tree))


def test_gradients_flow_into_every_loop_iteration(cfg, batch):
    """Every iteration of the shared block must contribute to its gradient."""
    tokens, t = batch
    params = init_params(cfg, mx.random.key(5))
    loop_count = 3
    nm = cfg.mid_layers

    def loss(p, count):
        return mx.sum(mx.square(forward(p, tokens, t, cfg, count).logits))

    looped = mx.grad(lambda p: loss(p, loop_count))(params)
    for g in _mid_leaf(looped):
        for _, array in mu.tree_flatten(g):
            assert bool(mx.all(mx.isfinite(array)).item())
    # every part of the shared block receives gradient
    for name in ("attn", "mlp", "ada", "norm1", "norm2"):
        for leaf_name, array in mu.tree_flatten(looped["mid"][0][name]):
            assert float(mx.max(mx.abs(array))) > 0.0, f"{name}.{leaf_name} has no gradient"

    # Making the R applications use *separate copies* of the mid-block exposes
    # each iteration's own contribution; with tied arrays MLX would merge them.
    separate = dict(params)
    separate["mid"] = [
        _deep_copy(layer) for _ in range(loop_count) for layer in params["mid"]
    ]
    assert separate["mid"][nm]["attn"]["wq"] is not params["mid"][0]["attn"]["wq"]
    copies = mx.grad(lambda p: loss(p, 1))(separate)
    per_iteration = []
    for i in range(loop_count * nm):
        mass = _grad_mass(copies["mid"][i])
        assert mass > 0.0, f"mid layer {i} (iteration {i // nm}) contributes nothing"
        per_iteration.append(mass)

    # ... and those contributions must add back up to the tied model's gradient,
    # which is the same statement as "the loop accounts for all R iterations".
    for i in range(nm):
        total = copies["mid"][i]
        for r in range(1, loop_count):
            total = mu.tree_map(lambda a, b: a + b, total, copies["mid"][r * nm + i])
        expected = dict(mu.tree_flatten(looped["mid"][i]))
        for name, array in mu.tree_flatten(total):
            assert mx.allclose(array, expected[name], atol=1e-6).item(), name


def test_gradients_are_finite_for_all_loop_counts(cfg, batch):
    tokens, t = batch
    params = init_params(cfg, mx.random.key(6))
    for loop_count in (1, 2, 4):
        grads = mx.grad(
            lambda p: mx.sum(mx.square(forward(p, tokens, t, cfg, loop_count).logits))
        )(params)
        flat = mu.tree_flatten(grads)
        assert all(bool(mx.all(mx.isfinite(a)).item()) for _, a in flat)
        assert any(float(mx.max(mx.abs(a)).item()) > 0 for _, a in flat)


def test_shapes_dtypes_and_mask_token(cfg, batch):
    tokens, t = batch
    params = init_params(cfg, mx.random.key(7))
    out = forward(params, tokens, t, cfg, loop_count=2)
    assert out.logits.shape == (3, cfg.seq_len, cfg.vocab_size)
    assert out.logits.dtype == mx.float32
    assert params["embed"].shape[0] == cfg.padded_vocab

    masked = mx.full((2, cfg.seq_len), cfg.mask_id, dtype=mx.int32)
    out_mask = forward(params, masked, 0.5, cfg, loop_count=2)
    assert out_mask.logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert bool(mx.all(mx.isfinite(out_mask.logits)).item())


def test_time_conditioning_accepts_scalars_and_changes_the_output(cfg, batch):
    tokens, _ = batch
    params = init_params(cfg, mx.random.key(8))
    a = forward(params, tokens, 0.2, cfg, loop_count=2)
    b = forward(params, tokens, 0.8, cfg, loop_count=2)
    assert not mx.allclose(a.logits, b.logits, atol=1e-6).item()
    scalar = forward(params, tokens, mx.array(0.2), cfg, loop_count=2)
    assert mx.array_equal(scalar.logits, a.logits).item()


def test_relative_hidden_change_hand_computed():
    a = mx.array([[0.0, 0.0]])
    b = mx.array([[3.0, 4.0]])
    assert float(relative_hidden_change(a, b)) == pytest.approx(5.0 / 5.0)
    assert float(relative_hidden_change(b, b)) == pytest.approx(0.0)


def test_adaptive_forward_bounds_and_agreement_with_the_fixed_forward(cfg, batch):
    tokens, t = batch
    params = init_params(cfg, mx.random.key(9))
    # a threshold that can never be met uses the whole budget
    full = forward_adaptive(params, tokens, t, cfg, max_loops=4, eps=0.0)
    assert full.loops_used == 4
    assert len(full.rel_changes) == 4
    assert mx.array_equal(
        full.logits, forward(params, tokens, t, cfg, loop_count=4).logits
    ).item()
    # a threshold that is always met stops after one iteration
    one = forward_adaptive(params, tokens, t, cfg, max_loops=4, eps=1e9)
    assert one.loops_used == 1
    assert mx.array_equal(
        one.logits, forward(params, tokens, t, cfg, loop_count=1).logits
    ).item()
    assert one.loops_used * cfg.mid_layers + cfg.head_layers + cfg.tail_layers == (
        layer_applications(cfg, 1)
    )


def test_adaptive_loop_count_is_monotone_in_the_threshold(cfg, batch):
    tokens, t = batch
    params = init_params(cfg, mx.random.key(10))
    for eps in (1e-9, 1e-4, 1e-2, 1.0):
        out = forward_adaptive(params, tokens, t, cfg, max_loops=6, eps=eps)
        assert 1 <= out.loops_used <= 6
    counts = [
        forward_adaptive(params, tokens, t, cfg, max_loops=6, eps=eps).loops_used
        for eps in (1e-9, 1e-4, 1e-2, 1.0)
    ]
    assert counts == sorted(counts, reverse=True)


def test_adaptive_capture_counts_blocks_of_the_used_loop_count(cfg, batch):
    tokens, t = batch
    params = init_params(cfg, mx.random.key(11))
    out = forward_adaptive(params, tokens, t, cfg, max_loops=5, eps=0.02, capture=True)
    assert out.n_blocks == layer_applications(cfg, out.loops_used)
    assert len(out.loop_states or []) == 0  # states are opt-in


def test_invalid_loop_counts_are_rejected(cfg, batch):
    tokens, t = batch
    params = init_params(cfg, mx.random.key(12))
    with pytest.raises(ValueError):
        forward(params, tokens, t, cfg, loop_count=0)
    with pytest.raises(ValueError):
        unroll_params(params, 0)
    with pytest.raises(ValueError):
        forward_adaptive(params, tokens, t, cfg, max_loops=0)
    with pytest.raises(ValueError):
        layer_applications(cfg, 0)
