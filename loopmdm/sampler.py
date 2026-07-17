"""Iterative unmasking samplers, with fixed or adaptive loop counts.

Two samplers share the denoising-step machinery:

``confidence_sample``
    the parallel-decoding rule used by the paper for evaluation: at each step the
    still-masked positions are ranked by the model's maximum token probability
    and the most confident ones are committed (Top-k unmasking).

``stochastic_sample``
    the formal reverse process of paper Eq. 2: a masked position stays masked
    with probability ``alpha_{s|t}`` and is otherwise drawn from the denoiser.

Both walk ``t`` from 1 to 0 and unmask according to the schedule's own marginal,
so the number of masked positions after each step matches the training-time
corruption process.  The loop count per denoising step is either fixed or chosen
by the paper's adaptive rule (stop when the relative hidden-state change between
two loop iterations drops below ``eps``).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .config import ModelConfig
from .model import forward, forward_adaptive, layer_applications
from .schedule import LINEAR, Schedule, sample_reverse_step


@dataclass
class SampleTrace:
    """What one sampling run did: the tokens and the compute it spent."""

    tokens: mx.array
    #: loop count used at each denoising step
    loop_counts: list
    #: number of still-masked positions per row after each step
    remaining_masked: list
    #: total transformer block evaluations (see ``layer_applications``)
    layer_applications: int
    #: relative hidden-state changes per loop iteration, adaptive runs only
    rel_changes: list | None = None

    @property
    def mean_loop_count(self) -> float:
        return sum(self.loop_counts) / max(len(self.loop_counts), 1)

    @property
    def loops_by_step(self) -> list:
        return list(self.loop_counts)


def time_grid(steps: int) -> list[float]:
    """``t`` from 1 down to 0 in ``steps`` equal intervals (inclusive endpoints)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return [1.0 - i / steps for i in range(steps + 1)]


def _denoise(params, x, t, cfg, loop_count, max_loops, eps):
    """One denoiser call; returns ``(logits, loops_used, rel_changes)``."""
    if max_loops is None:
        out = forward(params, x, t, cfg, loop_count=loop_count)
        return out.logits, loop_count, None
    out = forward_adaptive(params, x, t, cfg, max_loops=max_loops, eps=eps)
    return out.logits, out.loops_used, out.rel_changes


def _target_masked(n_fill: mx.array, alpha: float) -> mx.array:
    """How many positions should still be masked at time ``t`` (the forward marginal)."""
    return mx.round(n_fill.astype(mx.float32) * (1.0 - alpha)).astype(mx.int32)


def confidence_sample(
    params: dict,
    tokens: mx.array,
    cfg: ModelConfig,
    fill: mx.array,
    *,
    steps: int = 8,
    loop_count: int = 1,
    max_loops: int | None = None,
    eps: float = 0.1,
    schedule: Schedule = LINEAR,
    key: mx.array | None = None,
    sample_tokens: bool = False,
) -> SampleTrace:
    """Confidence-ordered parallel unmasking of the ``fill`` positions.

    ``tokens`` must already have ``cfg.mask_id`` at every ``fill`` position;
    positions outside ``fill`` are the prompt and are never modified.
    """
    grid = time_grid(steps)
    x = tokens
    loops: list[int] = []
    changes: list | None = [] if max_loops is not None else None
    remaining: list = []
    n_fill = mx.sum(fill.astype(mx.int32), axis=1)
    total_blocks = 0
    keys = mx.random.split(key, steps) if (sample_tokens and key is not None) else None

    for i in range(1, steps + 1):
        t_prev, t_next = grid[i - 1], grid[i]
        logits, used, rels = _denoise(params, x, t_prev, cfg, loop_count, max_loops, eps)
        loops.append(int(used))
        total_blocks += layer_applications(cfg, int(used))
        if changes is not None:
            changes.append(rels)

        still_masked = mx.logical_and(fill, mx.equal(x, cfg.mask_id))
        target = _target_masked(n_fill, float(schedule.alpha(t_next)))
        current = mx.sum(still_masked.astype(mx.int32), axis=1)
        k = mx.maximum(current - target, 0)

        probs = mx.softmax(logits, axis=-1)
        confidence = mx.max(probs, axis=-1)
        ranked = mx.argsort(-mx.where(still_masked, confidence, -1e30), axis=-1)
        rank = mx.argsort(ranked, axis=-1)
        chosen = mx.logical_and(still_masked, rank < k[:, None])

        if sample_tokens:
            logits_k = mx.log(mx.maximum(probs, 1e-30))
            drawn = mx.random.categorical(logits_k, key=keys[i - 1])
        else:
            drawn = mx.argmax(probs, axis=-1)
        x = mx.where(chosen, drawn.astype(x.dtype), x)
        mx.eval(x)
        remaining.append(mx.sum(mx.logical_and(fill, mx.equal(x, cfg.mask_id)), axis=1).tolist())

    return SampleTrace(
        tokens=x,
        loop_counts=loops,
        remaining_masked=remaining,
        layer_applications=total_blocks,
        rel_changes=changes,
    )


def stochastic_sample(
    params: dict,
    tokens: mx.array,
    cfg: ModelConfig,
    fill: mx.array,
    *,
    steps: int = 8,
    loop_count: int = 1,
    max_loops: int | None = None,
    eps: float = 0.1,
    schedule: Schedule = LINEAR,
    key: mx.array,
) -> SampleTrace:
    """Sample from the reverse process of paper Eq. 2 (no confidence heuristic)."""
    grid = time_grid(steps)
    x = tokens
    loops: list[int] = []
    changes: list | None = [] if max_loops is not None else None
    remaining: list = []
    total_blocks = 0

    if key is None:
        raise ValueError("stochastic_sample needs a key")
    keys = mx.random.split(key, steps)
    for i in range(1, steps + 1):
        t_prev, t_next = grid[i - 1], grid[i]
        logits, used, rels = _denoise(params, x, t_prev, cfg, loop_count, max_loops, eps)
        loops.append(int(used))
        total_blocks += layer_applications(cfg, int(used))
        if changes is not None:
            changes.append(rels)

        probs = mx.softmax(logits, axis=-1)
        mask = mx.equal(x, cfg.mask_id)
        x = sample_reverse_step(x, mask, probs, t_next, t_prev, schedule, key=keys[i - 1])
        # the reverse process keeps unmasked tokens; re-assert the prompt anyway
        x = mx.where(fill, x, tokens)
        mx.eval(x)
        remaining.append(mx.sum(mx.logical_and(fill, mx.equal(x, cfg.mask_id)), axis=1).tolist())

    x = mx.where(mx.logical_and(fill, mx.equal(x, cfg.mask_id)), tokens, x)
    return SampleTrace(
        tokens=x,
        loop_counts=loops,
        remaining_masked=remaining,
        layer_applications=total_blocks,
        rel_changes=changes,
    )


def masked_prompt(tokens: mx.array, fill: mx.array, mask_id: int) -> mx.array:
    """Return ``tokens`` with every ``fill`` position replaced by the mask token."""
    return mx.where(fill, mx.full(tokens.shape, mask_id, dtype=tokens.dtype), tokens)
