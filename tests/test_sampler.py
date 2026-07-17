"""The samplers must unmask on the schedule and account for their compute."""

from __future__ import annotations

import mlx.core as mx
import pytest

from loopmdm.config import ModelConfig
from loopmdm.model import init_params, layer_applications
from loopmdm.sampler import (
    confidence_sample,
    masked_prompt,
    stochastic_sample,
    time_grid,
)
from loopmdm.schedule import LINEAR, Schedule


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=16, seq_len=8, d_model=32, n_heads=4, n_layers=4,
        head_layers=1, mid_layers=2, time_dim=32,
    )


@pytest.fixture
def setup(cfg):
    params = init_params(cfg, mx.random.key(0))
    tokens = mx.random.randint(0, cfg.vocab_size, shape=(3, cfg.seq_len), key=mx.random.key(1))
    fill = mx.zeros(tokens.shape, dtype=mx.bool_)
    fill = fill.at[:, cfg.seq_len - 2 :].add(True)
    return params, tokens, fill


def test_masked_prompt_only_masks_the_fill_positions(cfg, setup):
    _, tokens, fill = setup
    masked = masked_prompt(tokens, fill, cfg.mask_id)
    assert mx.array_equal(masked[:, : cfg.seq_len - 2], tokens[:, : cfg.seq_len - 2]).item()
    assert mx.all(mx.equal(masked[:, cfg.seq_len - 2 :], cfg.mask_id)).item()


def test_time_grid_spans_one_to_zero():
    assert time_grid(4) == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert time_grid(1) == [1.0, 0.0]
    with pytest.raises(ValueError):
        time_grid(0)


def test_confidence_sampler_fills_everything_and_keeps_the_prompt(cfg, setup):
    params, tokens, fill = setup
    trace = confidence_sample(params, masked_prompt(tokens, fill, cfg.mask_id), cfg, fill, steps=4)
    assert not mx.any(mx.logical_and(fill, mx.equal(trace.tokens, cfg.mask_id))).item()
    # the prompt is untouched and every fill position now holds a real token
    assert mx.array_equal(trace.tokens[:, : cfg.seq_len - 2], tokens[:, : cfg.seq_len - 2]).item()
    assert bool(mx.all(trace.tokens[:, cfg.seq_len - 2 :] < cfg.vocab_size).item())


@pytest.mark.parametrize("steps", [1, 4, 8])
def test_remaining_mask_count_follows_the_schedule(cfg, setup, steps):
    """The sampler's mask counts must track the training-time corruption marginal."""
    params, tokens, fill = setup
    trace = confidence_sample(
        params, masked_prompt(tokens, fill, cfg.mask_id), cfg, fill, steps=steps
    )
    n_fill = int(mx.sum(fill.astype(mx.int32), axis=1)[0])
    grid = time_grid(steps)
    for i, remaining in enumerate(trace.remaining_masked, start=1):
        expected = round(n_fill * (1.0 - float(LINEAR.alpha(grid[i]))))
        assert remaining == [expected] * 3


def test_schedule_consistency_holds_for_the_cosine_schedule(cfg, setup):
    params, tokens, fill = setup
    sch = Schedule("cosine")
    trace = confidence_sample(
        params, masked_prompt(tokens, fill, cfg.mask_id), cfg, fill, steps=6, schedule=sch
    )
    n_fill = int(mx.sum(fill.astype(mx.int32), axis=1)[0])
    for i, remaining in enumerate(trace.remaining_masked, start=1):
        expected = round(n_fill * (1.0 - float(sch.alpha(time_grid(6)[i]))))
        assert remaining == [expected] * 3


def test_layer_applications_are_counted_from_the_loop_counts(cfg, setup):
    params, tokens, fill = setup
    trace = confidence_sample(
        params, masked_prompt(tokens, fill, cfg.mask_id), cfg, fill,
        steps=5, loop_count=3,
    )
    assert len(trace.loop_counts) == 5
    assert trace.loop_counts == [3] * 5
    assert trace.layer_applications == sum(layer_applications(cfg, r) for r in trace.loop_counts)
    assert trace.layer_applications == 5 * (cfg.head_layers + 3 * cfg.mid_layers + cfg.tail_layers)
    assert trace.mean_loop_count == 3.0


def test_more_denoising_steps_cost_more_compute(cfg, setup):
    params, tokens, fill = setup
    masked = masked_prompt(tokens, fill, cfg.mask_id)
    few = confidence_sample(params, masked, cfg, fill, steps=2, loop_count=2)
    many = confidence_sample(params, masked, cfg, fill, steps=8, loop_count=2)
    assert many.layer_applications > few.layer_applications


def test_adaptive_loop_counts_respect_the_budget_and_the_threshold(cfg, setup):
    params, tokens, fill = setup
    masked = masked_prompt(tokens, fill, cfg.mask_id)
    tight = confidence_sample(params, masked, cfg, fill, steps=4, max_loops=4, eps=0.0)
    assert tight.loop_counts == [4] * 4
    assert len(tight.rel_changes) == 4
    assert all(len(c) == 4 for c in tight.rel_changes)

    loose = confidence_sample(params, masked, cfg, fill, steps=4, max_loops=4, eps=1e9)
    assert loose.loop_counts == [1] * 4
    assert loose.layer_applications == 4 * layer_applications(cfg, 1)

    mid = confidence_sample(params, masked, cfg, fill, steps=4, max_loops=4, eps=0.05)
    assert all(1 <= r <= 4 for r in mid.loop_counts)
    assert mid.layer_applications == sum(layer_applications(cfg, r) for r in mid.loop_counts)


def test_sampling_is_deterministic(cfg, setup):
    params, tokens, fill = setup
    masked = masked_prompt(tokens, fill, cfg.mask_id)
    a = confidence_sample(params, masked, cfg, fill, steps=4, loop_count=2)
    b = confidence_sample(params, masked, cfg, fill, steps=4, loop_count=2)
    assert mx.array_equal(a.tokens, b.tokens).item()
    assert a.loop_counts == b.loop_counts
    # ... and identical for the same key when tokens are drawn
    c = confidence_sample(params, masked, cfg, fill, steps=4, sample_tokens=True, key=mx.random.key(5))
    d = confidence_sample(params, masked, cfg, fill, steps=4, sample_tokens=True, key=mx.random.key(5))
    assert mx.array_equal(c.tokens, d.tokens).item()


def test_confidence_sampling_picks_the_most_confident_positions(cfg, setup):
    """With one unmask per step the committed position must be the argmax one."""
    params, tokens, fill = setup
    masked = masked_prompt(tokens, fill, cfg.mask_id)
    trace = confidence_sample(params, masked, cfg, fill, steps=2, loop_count=1)
    # after the first step exactly one of the two fill positions is committed
    assert trace.remaining_masked[0] == [1] * 3
    assert trace.remaining_masked[1] == [0] * 3


def test_stochastic_sampler_fills_positions_and_preserves_the_prompt(cfg, setup):
    params, tokens, fill = setup
    masked = masked_prompt(tokens, fill, cfg.mask_id)
    trace = stochastic_sample(params, masked, cfg, fill, steps=6, loop_count=2, key=mx.random.key(2))
    assert not mx.any(mx.logical_and(fill, mx.equal(trace.tokens, cfg.mask_id))).item()
    assert mx.array_equal(trace.tokens[:, : cfg.seq_len - 2], tokens[:, : cfg.seq_len - 2]).item()
    assert len(trace.loop_counts) == 6
    assert trace.layer_applications == 6 * layer_applications(cfg, 2)
    # the final step has alpha_{0|t} = 0, so nothing may remain masked
    assert trace.remaining_masked[-1] == [0] * 3


def test_stochastic_sampler_needs_a_key(cfg, setup):
    params, tokens, fill = setup
    with pytest.raises(ValueError):
        stochastic_sample(params, masked_prompt(tokens, fill, cfg.mask_id), cfg, fill, key=None)


def test_stochastic_sampler_never_unmasks_earlier_than_the_schedule(cfg, setup):
    params, tokens, fill = setup
    trace = stochastic_sample(
        params, masked_prompt(tokens, fill, cfg.mask_id), cfg, fill, steps=5, key=mx.random.key(3)
    )
    counts = [r[0] for r in trace.remaining_masked]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 0
