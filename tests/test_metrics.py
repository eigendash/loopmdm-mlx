"""Metrics must be hand-checkable, and the attention measure must be bounded."""

from __future__ import annotations

import mlx.core as mx
import pytest

from loopmdm.config import ModelConfig
from loopmdm.metrics import (
    answer_nll,
    loop_profile_by_timestep,
    mask_to_mask_attention,
    mid_attention_by_iteration,
    span_exact_match,
    span_token_accuracy,
    total_layer_applications,
)
from loopmdm.model import forward, init_params, layer_applications


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=12, seq_len=6, d_model=16, n_heads=4, n_layers=4,
        head_layers=1, mid_layers=2, time_dim=16,
    )


def test_mask_to_mask_attention_hand_computed():
    probs = [mx.array([[[[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.5, 0.1, 0.4]]]])]
    mask = mx.array([[True, False, True]])
    # (A[0,0] + A[0,2] + A[2,0] + A[2,2]) / 2 masked queries
    assert mask_to_mask_attention(probs, mask)[0] == pytest.approx((0.1 + 0.7 + 0.5 + 0.4) / 2)


def test_mask_to_mask_attention_is_one_when_everything_is_masked():
    probs = [mx.array([[[[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.5, 0.1, 0.4]]]])]
    mask = mx.ones((1, 3), dtype=mx.bool_)
    assert mask_to_mask_attention(probs, mask)[0] == pytest.approx(1.0)


def test_mask_to_mask_attention_zero_without_masked_queries():
    probs = [mx.array([[[[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.5, 0.1, 0.4]]]])]
    mask = mx.zeros((1, 3), dtype=mx.bool_)
    assert mask_to_mask_attention(probs, mask)[0] == 0.0
    assert mask_to_mask_attention([], mask) == []


def test_mask_to_mask_attention_is_bounded_and_averages_heads():
    rows = mx.softmax(mx.random.normal((2, 3, 4, 4), key=mx.random.key(0)), axis=-1)
    mask = mx.random.uniform(shape=(2, 4), key=mx.random.key(1)) < 0.5
    values = mask_to_mask_attention([rows, rows], mask)
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values[0] == values[1]
    # a single-head matrix with uniform attention gives the masked fraction
    uniform = mx.full((1, 1, 4, 4), 0.25)
    m = mx.array([[True, True, False, False]])
    assert mask_to_mask_attention([uniform], m)[0] == pytest.approx(0.5)


def test_answer_nll_hand_computed():
    logits = mx.array([[[0.0, 0.0], [0.0, 0.0]]])  # uniform over 2 tokens
    tokens = mx.array([[0, 1]])
    span = mx.array([[False, True]])
    # one answer position, uniform prediction -> NLL = log(2)
    assert answer_nll(logits, tokens, span) == pytest.approx(float(mx.log(mx.array(2.0))), rel=1e-5)


def test_answer_nll_uses_only_the_span():
    logits = mx.array([[[4.0, 0.0], [4.0, 0.0]]])
    tokens = mx.array([[1, 0]])
    span = mx.array([[False, True]])
    # logits strongly prefer token 0, the span token is 0 -> small NLL
    assert answer_nll(logits, tokens, span) < 0.05


def test_span_exact_match_and_token_accuracy():
    target = mx.array([[1, 2, 3], [4, 5, 6]])
    span = mx.array([[False, True, True], [False, True, True]])
    perfect = mx.array([[9, 2, 3], [9, 5, 6]])
    partial = mx.array([[9, 2, 7], [9, 5, 6]])
    assert span_exact_match(perfect, target, span) == 1.0
    assert span_exact_match(partial, target, span) == 0.5
    assert span_token_accuracy(partial, target, span) == pytest.approx(0.75)
    # mismatches outside the span do not count
    assert span_exact_match(mx.array([[0, 2, 3], [0, 5, 6]]), target, span) == 1.0


def test_mid_attention_by_iteration_groups_and_validates(cfg):
    params = init_params(cfg, mx.random.key(0))
    tokens = mx.random.randint(0, cfg.vocab_size, shape=(2, cfg.seq_len), key=mx.random.key(1))
    out = forward(params, tokens, 0.5, cfg, loop_count=3, capture=True)
    grouped = mid_attention_by_iteration(out.attentions, cfg, 3)
    assert len(grouped) == 3 and all(len(g) == cfg.mid_layers for g in grouped)
    # the first application is the same whatever the loop count is
    one = forward(params, tokens, 0.5, cfg, loop_count=1, capture=True)
    for a, b in zip(grouped[0], one.attentions[cfg.head_layers :]):
        assert mx.array_equal(a, b).item()
    with pytest.raises(ValueError):
        mid_attention_by_iteration(out.attentions, cfg, 2)


def test_total_layer_applications_matches_the_model(cfg):
    assert total_layer_applications(cfg, [1, 2, 3]) == (
        layer_applications(cfg, 1) + layer_applications(cfg, 2) + layer_applications(cfg, 3)
    )


def test_loop_profile_by_timestep_buckets():
    assert loop_profile_by_timestep([1, 1, 4, 4], 4, buckets=2) == [1.0, 4.0]
    assert loop_profile_by_timestep([2, 2, 2, 2], 4, buckets=4) == [2.0] * 4
    with pytest.raises(ValueError):
        loop_profile_by_timestep([1, 2, 3], 4)
