"""The training objective: masking, weighting, loop sampling, optimisation."""

from __future__ import annotations

import mlx.core as mx
import mlx.utils as mu
import pytest

from loopmdm.config import ModelConfig, TrainConfig
from loopmdm.model import count_params, forward, init_params, num_parameters
from loopmdm.schedule import LINEAR, Schedule
from loopmdm.train import (
    clip_gradients,
    lr_at,
    mdm_loss,
    sample_loop_count,
    train_mdm,
)


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=16, seq_len=8, d_model=32, n_heads=4, n_layers=3,
        head_layers=1, mid_layers=1, time_dim=32,
    )


@pytest.fixture
def batch(cfg):
    tokens = mx.random.randint(0, cfg.vocab_size, shape=(4, cfg.seq_len), key=mx.random.key(0))
    t = mx.array([0.2, 0.5, 0.8, 0.35])
    return tokens, t


def test_loss_ignores_unmasked_positions(cfg, batch):
    tokens, t = batch
    params = init_params(cfg, mx.random.key(1))
    mask = mx.zeros(tokens.shape, dtype=mx.bool_)
    out = mdm_loss(params, tokens, cfg, LINEAR, t=t, mask=mask, loop_count=1)
    assert float(out.loss) == 0.0


def test_loss_is_the_mean_cross_entropy_over_masked_positions(cfg, batch):
    tokens, t = batch
    params = init_params(cfg, mx.random.key(2))
    mask = mx.random.uniform(shape=tokens.shape, key=mx.random.key(3)) < 0.5
    out = mdm_loss(params, tokens, cfg, LINEAR, t=t, mask=mask, loop_count=2)

    ref = forward(params, out.x_t, t, cfg, loop_count=2).logits
    logp = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    target = mx.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]
    expected = -mx.sum(mx.where(mask, target, mx.zeros_like(target))) / mx.sum(mask)
    assert float(out.loss) == pytest.approx(float(expected), rel=1e-5)


@pytest.mark.parametrize("name", ["linear", "cosine"])
def test_nelbo_weighting_matches_the_analytic_weight(cfg, batch, name):
    tokens, t = batch
    params = init_params(cfg, mx.random.key(4))
    mask = mx.random.uniform(shape=tokens.shape, key=mx.random.key(5)) < 0.5
    sch = Schedule(name)
    out = mdm_loss(
        params, tokens, cfg, sch, t=t, mask=mask, loop_count=1, weighting="nelbo"
    )
    ref = forward(params, out.x_t, t, cfg, loop_count=1).logits
    logp = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    target = mx.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]
    w = mask.astype(mx.float32) * sch.nelbo_weight(t).reshape(-1, 1)
    expected = -mx.sum(w * target) / mx.sum(w)
    assert float(out.loss) == pytest.approx(float(expected), rel=1e-5)
    # the weight really is 1/t for the linear schedule
    if name == "linear":
        assert mx.allclose(w, mask.astype(mx.float32) / t.reshape(-1, 1)).item()


def test_reported_masked_fraction_matches_t(cfg):
    tokens = mx.zeros((4000, 1), dtype=mx.int32)
    params = init_params(cfg, mx.random.key(6))
    t = mx.full((4000,), 0.4)
    out = mdm_loss(params, tokens, cfg, LINEAR, t=t, key=mx.random.key(7), loop_count=1)
    assert float(out.masked_fraction) == pytest.approx(0.4, abs=0.02)


def test_loop_count_sampling_is_uniform_over_the_range():
    counts = [sample_loop_count(4, mx.random.key(i)) for i in range(4000)]
    assert set(counts) == {1, 2, 3, 4}
    assert sum(counts) / len(counts) == pytest.approx(2.5, abs=0.06)
    assert sample_loop_count(1, mx.random.key(0)) == 1
    with pytest.raises(ValueError):
        sample_loop_count(0, mx.random.key(0))


def test_loss_uses_the_sampled_loop_count(cfg, batch):
    tokens, _ = batch
    params = init_params(cfg, mx.random.key(8))
    seen = set()
    for i in range(60):
        out = mdm_loss(params, tokens, cfg, LINEAR, mx.random.key(i), max_loops=3)
        seen.add(out.loop_count)
    assert seen == {1, 2, 3}


def _easy_corpus(cfg):
    """Every position has a fixed token: the denoiser only has to learn to place it."""
    return mx.broadcast_to(mx.arange(1, cfg.seq_len + 1), (8, cfg.seq_len))


def test_training_reduces_loss_on_a_learnable_corpus(cfg):
    corpus = _easy_corpus(cfg)
    tc = TrainConfig(steps=200, batch_size=8, lr=3e-3, warmup=10, seed=0, max_loops=1)
    result = train_mdm(cfg, tc, corpus, verbose=False)
    first = sum(h["loss"] for h in result.history[:10]) / 10
    last = sum(h["loss"] for h in result.history[-10:]) / 10
    assert last < 0.4 * first, (first, last)
    assert last < 1.0


def test_training_with_loops_still_learns(cfg):
    """The looped model must train through the shared block (gradients reach it)."""
    corpus = _easy_corpus(cfg)
    tc = TrainConfig(steps=300, batch_size=8, lr=3e-3, warmup=10, seed=0, max_loops=3)
    result = train_mdm(cfg, tc, corpus, verbose=False)
    first = sum(h["loss"] for h in result.history[:10]) / 10
    last = sum(h["loss"] for h in result.history[-10:]) / 10
    assert last < 0.5 * first, (first, last)
    assert count_params(result.params) == num_parameters(cfg)
    assert set(h["loop_count"] for h in result.history) == {1, 2, 3}


def test_training_is_reproducible(cfg):
    """Same seed -> same batches, times, masks, loop counts and (up to fp noise) weights.

    The data pipeline is exactly reproducible because every draw comes from an
    explicit key.  The weights agree only to float32 tolerance: some GPU
    reductions are not bit-deterministic in MLX.
    """
    corpus = _easy_corpus(cfg)
    tc = TrainConfig(steps=25, batch_size=8, warmup=5, seed=3, max_loops=2)
    a = train_mdm(cfg, tc, corpus, verbose=False)
    b = train_mdm(cfg, tc, corpus, verbose=False)
    assert [h["loop_count"] for h in a.history] == [h["loop_count"] for h in b.history]
    assert [h["masked_fraction"] for h in a.history] == [h["masked_fraction"] for h in b.history]
    for ha, hb in zip(a.history, b.history):
        assert ha["loss"] == pytest.approx(hb["loss"], rel=1e-4)
    for (n1, x1), (n2, x2) in zip(mu.tree_flatten(a.params), mu.tree_flatten(b.params)):
        assert n1 == n2
        assert mx.allclose(x1, x2, rtol=1e-3, atol=1e-4).item()


def test_lr_schedule_warms_up_then_decays():
    tc = TrainConfig(steps=100, lr=1e-2, warmup=10)
    warm = [lr_at(s, tc) for s in range(10)]
    assert warm == sorted(warm)
    assert lr_at(0, tc) < lr_at(9, tc)
    assert lr_at(10, tc) == pytest.approx(1e-2)
    assert lr_at(99, tc) == pytest.approx(1e-3, rel=0.05)
    assert lr_at(50, tc) < lr_at(10, tc)


def test_clip_gradients_bounds_the_global_norm():
    grads = {"a": mx.array([3.0, 4.0]), "b": [mx.array([0.0])]}
    clipped, norm = clip_gradients(grads, max_norm=1.0)
    assert float(norm) == pytest.approx(5.0)
    assert float(mx.sqrt(sum(mx.sum(mx.square(a)) for _, a in mu.tree_flatten(clipped)))) == pytest.approx(1.0, rel=1e-5)
    untouched, norm2 = clip_gradients(grads, max_norm=10.0)
    assert float(norm2) == pytest.approx(5.0)
    assert mx.array_equal(untouched["a"], grads["a"]).item()


@pytest.mark.parametrize("max_loops", [1, 3])
def test_loss_gradients_are_finite(cfg, batch, max_loops):
    tokens, _ = batch
    params = init_params(cfg, mx.random.key(12))
    grads = mx.grad(
        lambda p: mdm_loss(p, tokens, cfg, LINEAR, mx.random.key(13), max_loops=max_loops).loss
    )(params)
    leaves = mu.tree_flatten(grads)
    assert all(bool(mx.all(mx.isfinite(a)).item()) for _, a in leaves)
    assert any(float(mx.max(mx.abs(a)).item()) > 0 for _, a in leaves)


def test_loss_requires_a_key_or_explicit_masking(cfg, batch):
    tokens, _ = batch
    params = init_params(cfg, mx.random.key(14))
    with pytest.raises(ValueError):
        mdm_loss(params, tokens, cfg, LINEAR, None)
