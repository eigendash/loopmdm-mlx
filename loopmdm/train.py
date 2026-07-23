"""Training objective and optimisation loop.

The objective is the masked-diffusion NELBO of paper Eq. 3/5: sample a
timestep, corrupt the clean sequence, and take the cross-entropy of the model's
prediction at the masked positions only, optionally weighted by the
continuous-time weight ``-alpha'(t) / (1 - alpha_t)``.  During training the loop
count is sampled uniformly from ``{1, ..., S_max}`` at every step (paper Eq. 5),
so the shared mid-block sees a range of effective depths and the model can be
run at any loop count at inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, NamedTuple

import mlx.core as mx
import mlx.optimizers as optim
import mlx.utils as mu

from .config import ModelConfig, TrainConfig
from .model import forward, init_params
from .schedule import LINEAR, Schedule, corrupt_tokens, masked_weight


class MDMLoss(NamedTuple):
    loss: mx.array
    logits: mx.array
    mask: mx.array
    x_t: mx.array
    t: mx.array
    loop_count: int
    masked_fraction: mx.array


def sample_loop_count(max_loops: int, key: mx.array) -> int:
    """Uniform loop count in ``{1, ..., max_loops}`` (paper Eq. 5, one per step)."""
    if max_loops < 1:
        raise ValueError("max_loops must be >= 1")
    if max_loops == 1:
        return 1
    return int(mx.random.randint(1, max_loops + 1, shape=(1,), key=key).item())


def mdm_loss(
    params: dict,
    tokens: mx.array,
    cfg: ModelConfig,
    schedule: Schedule = LINEAR,
    key: mx.array | None = None,
    *,
    weighting: str = "uniform",
    loop_count: int | None = None,
    max_loops: int = 1,
    mask: mx.array | None = None,
    t: mx.array | None = None,
    ignore: mx.array | None = None,
    always: mx.array | None = None,
    realized_timestep: bool = False,
    t_max: float = 1.0,
) -> MDMLoss:
    """One Monte-Carlo estimate of the training objective.

    ``t`` and ``mask`` can be supplied to make the estimate deterministic in
    tests; otherwise they are sampled from ``key``.  ``ignore`` marks positions
    that are never corrupted and never contribute to the loss (padding) and
    ``always`` marks positions that are *always* corrupted (the answer span in
    the toy task, which keeps a clean gradient on the part being evaluated).
    ``realized_timestep`` feeds the model the realised mask rate instead of the
    sampled ``t``; combined with ``always`` this keeps the timestep consistent
    with what inference does.
    """
    if key is None and (mask is None or t is None):
        raise ValueError("mdm_loss needs a key or an explicit (t, mask)")
    batch = tokens.shape[0]
    keys = mx.random.split(key, 3) if key is not None else [None, None, None]
    if t is None:
        t = mx.random.uniform(shape=(batch,), key=keys[0]) * t_max
    if loop_count is None:
        loop_count = sample_loop_count(max_loops, keys[2]) if key is not None else 1
    x_t, mask = corrupt_tokens(tokens, t, cfg.mask_id, schedule, key=keys[1], mask=mask)
    if ignore is not None:
        mask = mx.logical_and(mask, mx.logical_not(ignore))
        x_t = mx.where(ignore, tokens, x_t)
    if always is not None:
        mask = mx.logical_or(mask, always)
        x_t = mx.where(always, mx.full(tokens.shape, cfg.mask_id, dtype=tokens.dtype), x_t)
    t_model = mx.mean(mask.astype(mx.float32), axis=1) if realized_timestep else t

    out = forward(params, x_t, t_model, cfg, loop_count=loop_count)
    log_probs = out.logits - mx.logsumexp(out.logits, axis=-1, keepdims=True)
    target = mx.take_along_axis(log_probs, tokens[..., None], axis=-1)[..., 0]
    weights = masked_weight(mask, t, schedule, weighting)
    total = mx.sum(weights)
    loss = -mx.sum(weights * target) / mx.maximum(total, 1.0)
    return MDMLoss(
        loss=loss,
        logits=out.logits,
        mask=mask,
        x_t=x_t,
        t=t,
        loop_count=loop_count,
        masked_fraction=mx.mean(mask.astype(mx.float32)),
    )


@dataclass
class TrainResult:
    params: dict
    history: list[dict]
    steps: int


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by a cosine decay to 10% of the peak rate."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(cfg.warmup, 1)
    progress = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def clip_gradients(grads: dict, max_norm: float) -> tuple[dict, mx.array]:
    """Rescale the gradient tree so its global L2 norm is at most ``max_norm``."""
    leaves = mu.tree_flatten(grads)
    total = mx.zeros(())
    for _, array in leaves:
        total = total + mx.sum(mx.square(array))
    norm = mx.sqrt(total)
    scale = mx.minimum(1.0, max_norm / mx.maximum(norm, 1e-6))
    return mu.tree_map(lambda a: a * scale, grads), norm


def train_mdm(
    cfg: ModelConfig,
    train_cfg: TrainConfig,
    corpus: mx.array,
    schedule: Schedule = LINEAR,
    key: mx.array | None = None,
    *,
    verbose: bool = True,
    log_every: int = 100,
    on_step: Callable[[int, dict], None] | None = None,
    ignore: mx.array | None = None,
    always: mx.array | None = None,
    realized_timestep: bool = False,
    t_max: float | None = None,
) -> TrainResult:
    """Train a (possibly looped) masked diffusion denoiser on a token corpus.

    ``corpus`` is an integer array of shape ``(N, seq_len)``; batches are drawn
    with replacement.  Reproducible given ``key``/``train_cfg.seed``.
    """
    if key is None:
        key = mx.random.key(train_cfg.seed)
    if t_max is None:
        t_max = train_cfg.t_max
    params = init_params(cfg, key)
    optimizer = optim.AdamW(
        learning_rate=lambda step: mx.array(lr_at(int(step), train_cfg)),
        weight_decay=train_cfg.weight_decay,
    )
    key, data_key, loss_key = mx.random.split(key, 3)
    history: list[dict] = []

    for step in range(train_cfg.steps):
        idx = mx.random.randint(0, corpus.shape[0], shape=(train_cfg.batch_size,), key=data_key)
        batch = corpus[idx]
        data_key, step_key = mx.random.split(data_key)

        def step_loss(p, batch=batch, step_key=step_key, step=step):
            out = mdm_loss(
                p,
                batch,
                cfg,
                schedule,
                step_key,
                weighting=train_cfg.loss_weighting,
                max_loops=train_cfg.max_loops,
                ignore=None if ignore is None else ignore[idx],
                always=None if always is None else always[idx],
                realized_timestep=realized_timestep,
                t_max=t_max,
            )
            lr = lr_at(step, train_cfg)
            return out.loss, (out.loop_count, out.masked_fraction, lr)

        (loss, aux), grads = mx.value_and_grad(step_loss)(params)
        loop_count, masked_fraction, lr = aux
        grads, grad_norm = clip_gradients(grads, train_cfg.grad_clip)
        params = optimizer.apply_gradients(grads, params)
        mx.eval(params, loss, grad_norm)
        record = {
            "step": step,
            "loss": float(loss.item()),
            "lr": float(lr),
            "grad_norm": float(grad_norm.item()),
            "masked_fraction": float(masked_fraction.item()),
            "loop_count": int(loop_count),
        }
        history.append(record)
        if on_step is not None:
            on_step(step, record)
        if verbose and (step % log_every == 0 or step == train_cfg.steps - 1):
            print(
                f"[train] step {step:5d}/{train_cfg.steps} loss {record['loss']:.4f} "
                f"lr {record['lr']:.2e} loops {record['loop_count']} "
                f"masked {record['masked_fraction']:.3f} grad {record['grad_norm']:.2f}",
                flush=True,
            )
    return TrainResult(params=params, history=history, steps=train_cfg.steps)
