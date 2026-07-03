"""The forward corruption process and the reverse transition must agree."""

from __future__ import annotations

import mlx.core as mx
import pytest

from loopmdm.schedule import (
    LINEAR,
    Schedule,

    corrupt_tokens,
    masked_weight,
    reverse_transition_probs,
    sample_reverse_step,
)

SCHEDULES = [Schedule("linear"), Schedule("cosine")]


@pytest.mark.parametrize("sch", SCHEDULES)
def test_alpha_endpoints_and_monotone(sch):
    t = mx.linspace(0.0, 1.0, 101)
    a = sch.alpha(t)
    assert abs(float(sch.alpha(0.0)) - 1.0) < 1e-6
    assert abs(float(sch.alpha(1.0))) < 1e-6
    diffs = mx.diff(a)
    assert float(mx.max(diffs)) < 1e-6, "alpha must be non-increasing"
    assert float(sch.mask_probability(0.0)) == pytest.approx(0.0, abs=1e-6)
    assert float(sch.mask_probability(1.0)) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("sch", SCHEDULES)
def test_dalpha_matches_finite_differences(sch):
    t = mx.array([0.1, 0.3, 0.5, 0.7, 0.9])
    eps = 1e-3
    fd = (sch.alpha(t + eps) - sch.alpha(t - eps)) / (2 * eps)
    assert mx.allclose(fd, sch.dalpha(t), atol=1e-3).item()


@pytest.mark.parametrize("sch", SCHEDULES)
def test_nelbo_weight_is_the_continuous_time_weight(sch):
    """The defining identity: w(t) * (1 - alpha_t) = -alpha'(t)."""
    t = mx.array([0.05, 0.2, 0.5, 0.9, 1.0])
    w = sch.nelbo_weight(t)
    lhs = w * (1.0 - sch.alpha(t))
    rhs = -sch.dalpha(t)
    assert mx.allclose(lhs, rhs, atol=1e-5).item()
    assert float(mx.min(w)) > 0


def test_linear_nelbo_weight_is_one_over_t():
    """For alpha_t = 1 - t the weight collapses to the familiar 1/t."""
    t = mx.array([0.1, 0.25, 0.5, 1.0])
    assert mx.allclose(LINEAR.nelbo_weight(t), 1.0 / t, atol=1e-5).item()


def test_nelbo_weight_is_clipped_near_zero():
    sch = Schedule("linear", t_min=0.05)
    assert float(sch.nelbo_weight(0.0)) == pytest.approx(1.0 / 0.05, rel=1e-5)


def test_alpha_ratio_hand_values():
    # linear: (1 - alpha_s) / (1 - alpha_t) = s / t
    assert float(LINEAR.alpha_ratio(0.25, 0.5)) == pytest.approx(0.5, rel=1e-6)
    assert float(LINEAR.alpha_ratio(0.0, 0.5)) == pytest.approx(0.0, abs=1e-6)
    assert float(LINEAR.alpha_ratio(0.5, 0.5)) == pytest.approx(1.0, rel=1e-6)
    # cosine at s = 0.5, t = 1: (1 - cos(pi/4)) / 1
    expected = 1.0 - float(mx.cos(mx.array(mx.pi / 4.0)))
    assert float(Schedule("cosine").alpha_ratio(0.5, 1.0)) == pytest.approx(
        expected, rel=1e-6
    )


def test_corrupt_endpoints():
    tokens = mx.arange(24).reshape(2, 12)
    x0, m0 = corrupt_tokens(tokens, 0.0, mask_id=99, key=mx.random.key(0))
    assert not m0.any().item()
    assert mx.array_equal(x0, tokens).item()
    x1, m1 = corrupt_tokens(tokens, 1.0, mask_id=99, key=mx.random.key(0))
    assert m1.all().item()
    assert mx.array_equal(x1, mx.full(tokens.shape, 99, dtype=tokens.dtype)).item()


@pytest.mark.parametrize("sch", SCHEDULES)
def test_empirical_corruption_rate_matches_schedule(sch):
    tokens = mx.zeros((20000, 1), dtype=mx.int32)
    for t in (0.25, 0.5, 0.75):
        _, mask = corrupt_tokens(tokens, t, mask_id=1, schedule=sch, key=mx.random.key(0))
        rate = float(mask.astype(mx.float32).mean())
        assert rate == pytest.approx(float(sch.mask_probability(t)), abs=0.015)


def test_corruption_is_reproducible_and_shape_preserving():
    tokens = mx.arange(6, dtype=mx.int32).reshape(2, 3)
    a = corrupt_tokens(tokens, 0.5, mask_id=7, key=mx.random.key(3))
    b = corrupt_tokens(tokens, 0.5, mask_id=7, key=mx.random.key(3))
    c = corrupt_tokens(tokens, 0.5, mask_id=7, key=mx.random.key(4))
    assert mx.array_equal(a[0], b[0]).item()
    assert not mx.array_equal(a[0], c[0]).item()
    assert a[0].shape == tokens.shape
    assert a[0].dtype == tokens.dtype


def test_corruption_accepts_an_explicit_mask():
    tokens = mx.arange(4, dtype=mx.int32)
    mask = mx.array([True, False, True, False])
    x_t, out = corrupt_tokens(tokens, 0.5, mask_id=9, mask=mask)
    assert mx.array_equal(out, mask).item()
    assert x_t.tolist() == [9, 1, 9, 3]


def test_masked_weight_selects_masked_positions():
    mask = mx.array([[True, False, True]])
    assert masked_weight(mask, 0.5).tolist() == [[1.0, 0.0, 1.0]]
    assert masked_weight(mask, 0.5, weighting="nelbo").tolist() == [[2.0, 0.0, 2.0]]
    with pytest.raises(ValueError):
        masked_weight(mask, 0.5, weighting="nope")


def test_reverse_transition_is_a_distribution_and_keeps_unmasked_tokens():
    vocab = 5
    tokens_t = mx.array([[1, vocab, 3, vocab]])  # mask_id == vocab
    mask = mx.array([[False, True, False, True]])
    probs = mx.array([[[1.0, 0, 0, 0, 0], [0.2, 0.3, 0.5, 0, 0],
                       [0, 0, 0, 1.0, 0], [0, 0, 1.0, 0, 0]]]).astype(mx.float32)
    out = reverse_transition_probs(tokens_t, mask, probs, 0.25, 0.5)  # ratio = 0.5
    assert out.shape == (1, 4, vocab + 1)
    assert mx.allclose(mx.sum(out, axis=-1), mx.ones((1, 4)), atol=1e-6).item()
    # position 0 is unmasked: point mass on token 1
    assert out[0, 0].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # position 1 is masked: 0.5 on the mask, 0.5 * p_theta on the real tokens
    assert out[0, 1].tolist() == pytest.approx([0.1, 0.15, 0.25, 0.0, 0.0, 0.5])
    # position 3 is masked and the denoiser is certain: 0.5 mask / 0.5 token 2
    assert out[0, 3].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.0, 0.0, 0.5])


def test_reverse_transition_is_identity_at_equal_times():
    vocab = 3
    tokens_t = mx.array([[vocab, 1]])
    mask = mx.array([[True, False]])
    probs = mx.full((1, 2, vocab), 1.0 / vocab)
    out = reverse_transition_probs(tokens_t, mask, probs, 0.6, 0.6)
    assert out[0, 0].tolist() == pytest.approx([0, 0, 0, 1.0])
    assert out[0, 1].tolist() == [0.0, 1.0, 0.0, 0.0]


def _oracle_probs(x0: mx.array, vocab: int) -> mx.array:
    """A denoiser that always predicts the clean sequence exactly."""
    return mx.eye(vocab)[x0]


@pytest.mark.parametrize("sch", SCHEDULES)
def test_oracle_reverse_step_preserves_the_forward_marginal(sch):
    """With a perfect denoiser, one reverse step reproduces q(x_s | x_0)."""
    vocab = 4
    n = 20000
    x0 = mx.random.randint(0, vocab, shape=(n, 1), key=mx.random.key(1))
    x_t, mask = corrupt_tokens(x0, 1.0, mask_id=vocab, schedule=sch, mask=mx.ones((n, 1), dtype=mx.bool_))
    assert mask.all().item()
    probs = _oracle_probs(x0, vocab)
    s = 0.3
    x_s = sample_reverse_step(
        x_t, mask, probs, s, 1.0, sch, key=mx.random.key(2)
    )
    masked_rate = float(mx.equal(x_s, vocab).astype(mx.float32).mean())
    assert masked_rate == pytest.approx(1.0 - float(sch.alpha(s)), abs=0.015)
    # unmasked positions carry the clean token
    unmasked = mx.logical_not(mx.equal(x_s, vocab))
    assert mx.array_equal(mx.where(unmasked, x_s, x0), x0).item()


@pytest.mark.parametrize("sch", SCHEDULES)
def test_oracle_reverse_chain_preserves_the_forward_marginal(sch):
    """A multi-step reverse chain from t=1 hits every intermediate marginal."""
    vocab = 4
    n = 20000
    x0 = mx.random.randint(0, vocab, shape=(n, 1), key=mx.random.key(5))
    probs = _oracle_probs(x0, vocab)
    x = mx.full((n, 1), vocab, dtype=mx.int32)
    mask = mx.ones((n, 1), dtype=mx.bool_)
    times = [1.0, 0.7, 0.4, 0.15, 0.0]
    for step, (t, s) in enumerate(zip(times[:-1], times[1:])):
        x = sample_reverse_step(x, mask, probs, s, t, sch, key=mx.random.key(100 + step))
        mask = mx.equal(x, vocab)
        expected = 1.0 - float(sch.alpha(s))
        assert float(mask.astype(mx.float32).mean()) == pytest.approx(expected, abs=0.02)
    # the final state is fully unmasked and equals the clean sequence
    assert mx.array_equal(x, x0).item()


def test_greedy_reverse_step_unmasks_all_at_s_zero():
    vocab = 3
    x_t = mx.full((2, 2), vocab, dtype=mx.int32)
    mask = mx.ones((2, 2), dtype=mx.bool_)
    probs = mx.eye(vocab)[mx.array([[2, 0], [1, 1]])]
    x_s = sample_reverse_step(x_t, mask, probs, 0.0, 1.0, LINEAR, key=mx.random.key(0), greedy=True)
    assert x_s.tolist() == [[2, 0], [1, 1]]


def test_alpha_ratio_rejects_bad_schedules():
    with pytest.raises(ValueError):
        Schedule("quadratic")
    with pytest.raises(ValueError):
        Schedule("linear", t_min=0.0)
