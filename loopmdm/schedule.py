"""The masked (absorbing-state) diffusion process.

Continuous-time masked diffusion, as used by the paper.  A schedule
``alpha_t`` decreases from ``alpha_0 = 1`` to ``alpha_1 = 0`` and each token of
a clean sequence ``x_0`` is independently corrupted by

    q(x_t^i | x_0^i) = alpha_t * delta_{x_0^i}(x_t^i) + (1 - alpha_t) * delta_m(x_t^i)

(paper Eq. 1).  The reverse process keeps already-unmasked tokens intact and
updates a masked position to a sample from the denoiser with probability
``1 - alpha_{s|t}`` where ``alpha_{s|t} = (1 - alpha_s) / (1 - alpha_t)``
(paper Eq. 2).  All functions here are pure and take an explicit ``key`` so
that corruption and sampling are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

_TINY = 1e-30


def _f32(t) -> mx.array:
    if isinstance(t, mx.array):
        return t.astype(mx.float32)
    return mx.array(float(t), dtype=mx.float32)


def _broadcast_rows(p: mx.array, ndim: int) -> mx.array:
    """Reshape a per-example ``(B,)`` tensor so it broadcasts over ``(B, T, ...)``."""
    if p.ndim == 1 and ndim > 1:
        return p.reshape(p.shape + (1,) * (ndim - 1))
    return p


@dataclass(frozen=True)
class Schedule:
    """Noise schedule ``alpha_t``.

    ``"linear"`` is ``alpha_t = 1 - t`` (the schedule used by MDLM/LLaDA and the
    default here); ``"cosine"`` is ``alpha_t = cos(pi t / 2)``.  Both satisfy
    ``alpha_0 = 1`` and ``alpha_1 = 0``.
    """

    name: str = "linear"
    # The NELBO weight -alpha'(t) / (1 - alpha(t)) diverges as t -> 0 for any
    # schedule with alpha_0 = 1, so t is clipped to [t_min, 1] when the exact
    # continuous-time weight is requested (only used by loss_weighting="nelbo").
    t_min: float = 0.02

    def __post_init__(self) -> None:
        if self.name not in ("linear", "cosine"):
            raise ValueError(f"unknown schedule {self.name!r}")
        if not 0.0 < self.t_min < 1.0:
            raise ValueError("t_min must lie in (0, 1)")

    def alpha(self, t) -> mx.array:
        t = _f32(t)
        if self.name == "linear":
            return 1.0 - t
        return mx.cos(0.5 * mx.pi * t)

    def dalpha(self, t) -> mx.array:
        """d alpha_t / dt."""
        t = _f32(t)
        if self.name == "linear":
            return mx.zeros_like(t) - 1.0
        return -0.5 * mx.pi * mx.sin(0.5 * mx.pi * t)

    def clip_t(self, t) -> mx.array:
        return mx.minimum(mx.maximum(_f32(t), self.t_min), 1.0)

    def mask_probability(self, t) -> mx.array:
        """P(x_t^i = m | x_0^i) = 1 - alpha_t."""
        return 1.0 - self.alpha(t)

    def alpha_ratio(self, s, t) -> mx.array:
        """``alpha_{s|t}``, the probability that a masked position stays masked."""
        num = 1.0 - self.alpha(s)
        den = 1.0 - self.alpha(t)
        return num / mx.maximum(den, _TINY)

    def nelbo_weight(self, t) -> mx.array:
        """The continuous-time NELBO weight ``-alpha'(t) / (1 - alpha_t)``."""
        t = self.clip_t(t)
        return -self.dalpha(t) / (1.0 - self.alpha(t))


LINEAR = Schedule("linear")


def corrupt_tokens(
    tokens: mx.array,
    t,
    mask_id: int,
    schedule: Schedule = LINEAR,
    *,
    key: mx.array | None = None,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Sample ``x_t ~ q(x_t | x_0)``.

    Returns ``(x_t, mask)`` where ``mask`` is the boolean indicator of the
    masked positions.  Either pass ``key`` (corruption is then random) or an
    explicit boolean ``mask``.
    """
    p = _broadcast_rows(schedule.mask_probability(t), tokens.ndim)
    if mask is None:
        if key is None:
            raise ValueError("corrupt_tokens needs either a key or an explicit mask")
        u = mx.random.uniform(shape=tokens.shape, key=key, dtype=mx.float32)
        mask = u < p
    x_t = mx.where(mask, mx.full(tokens.shape, mask_id, dtype=tokens.dtype), tokens)
    return x_t, mask


def masked_weight(
    mask: mx.array, t, schedule: Schedule = LINEAR, weighting: str = "uniform"
) -> mx.array:
    """Per-position loss weight: ``I[x_t = m]`` times the optional NELBO weight."""
    ind = mask.astype(mx.float32)
    if weighting == "uniform":
        return ind
    if weighting == "nelbo":
        w = _broadcast_rows(schedule.nelbo_weight(t), mask.ndim)
        return ind * w
    raise ValueError(f"unknown weighting {weighting!r}")


def reverse_transition_probs(
    tokens_t: mx.array,
    mask: mx.array,
    clean_probs: mx.array,
    s,
    t,
    schedule: Schedule = LINEAR,
) -> mx.array:
    """Distribution of ``x_s`` given ``x_t`` and the denoiser's ``p_theta(x_0|x_t)``.

    The result is a distribution over the padded vocabulary ``[V real tokens,
    mask]`` with the mask last, so ``mask_id == vocab_size == clean_probs.shape[-1]``.
    Unmasked positions are a point mass on their current token; a masked
    position stays masked with probability ``alpha_{s|t}`` and otherwise takes a
    token drawn from ``clean_probs`` (paper Eq. 2).
    """
    vocab = clean_probs.shape[-1]
    a = _broadcast_rows(schedule.alpha_ratio(s, t), mask.ndim)
    pad = mx.zeros(clean_probs.shape[:-1] + (1,), dtype=clean_probs.dtype)
    base = mx.concatenate([clean_probs, pad], axis=-1)
    onehot = mx.equal(tokens_t[..., None], mx.arange(vocab + 1))
    onehot = onehot.astype(clean_probs.dtype)
    return mx.where(mask[..., None], a * onehot + (1.0 - a) * base, onehot)


def sample_reverse_step(
    tokens_t: mx.array,
    mask: mx.array,
    clean_probs: mx.array,
    s,
    t,
    schedule: Schedule = LINEAR,
    *,
    key: mx.array,
    greedy: bool = False,
) -> mx.array:
    """One reverse step: draw ``x_s`` from ``reverse_transition_probs``."""
    probs = reverse_transition_probs(tokens_t, mask, clean_probs, s, t, schedule)
    if greedy:
        return mx.argmax(probs, axis=-1).astype(tokens_t.dtype)
    logits = mx.log(mx.maximum(probs, _TINY))
    return mx.random.categorical(logits, key=key).astype(tokens_t.dtype)
