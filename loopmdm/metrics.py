"""Evaluation helpers: answer accuracy/NLL, compute accounting, attention analysis.

The mask-to-mask measurement follows the paper's Appendix C.4: average the
attention mass that masked-position queries put on masked-position keys,

    M2M = mean_{masked queries} sum_{masked keys} A[q, k],

averaged over heads and batch elements.  Row-normalised attention makes 1.0 the
maximum (all of a masked query's mass on masked keys) and 0.0 the minimum.
"""

from __future__ import annotations

import mlx.core as mx

from .config import ModelConfig
from .model import layer_applications


def mask_to_mask_attention(attentions: list, mask: mx.array, indices=None) -> list:
    """Mask-to-mask attention for the selected attention matrices.

    ``attentions`` holds ``(B, H, T, T)`` probability matrices (``capture=True``
    on a forward pass); one float is returned per selected entry.
    """
    if not attentions:
        return []
    chosen = range(len(attentions)) if indices is None else indices
    mask_f = mask.astype(mx.float32)
    n_masked = mx.sum(mask_f, axis=1)  # (B,)
    total_queries = mx.maximum(mx.sum(n_masked), 1e-6)  # scalar
    out = []
    for i in chosen:
        probs = attentions[i]
        pair = mask_f[:, None, :, None] * mask_f[:, None, None, :]
        mass = mx.sum(probs * pair, axis=(-1, -2))  # (B, H)
        denom = probs.shape[1] * total_queries  # H * total masked queries
        out.append(float(mx.sum(mass) / denom))
    return out


def mid_attention_by_iteration(attentions: list, cfg: ModelConfig, loop_count: int) -> list:
    """Group the mid-block attention matrices as ``[loop iteration][mid layer]``."""
    expected = layer_applications(cfg, loop_count)
    if len(attentions) != expected:
        raise ValueError(f"expected {expected} attention matrices, got {len(attentions)}")
    nm = cfg.mid_layers
    mid = attentions[cfg.head_layers : cfg.head_layers + loop_count * nm]
    return [mid[k * nm : (k + 1) * nm] for k in range(loop_count)]


def answer_nll(logits: mx.array, tokens: mx.array, span: mx.array) -> float:
    """Mean negative log-likelihood over the positions selected by ``span``."""
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target = mx.take_along_axis(log_probs, tokens[..., None], axis=-1)[..., 0]
    weights = span.astype(mx.float32)
    total = mx.maximum(mx.sum(weights), 1e-6)
    return float(-mx.sum(weights * target) / total)


def span_exact_match(predicted: mx.array, target: mx.array, span: mx.array) -> float:
    """Fraction of rows where every position inside ``span`` matches."""
    agree = mx.logical_or(mx.equal(predicted, target), mx.logical_not(span))
    per_row = mx.all(agree, axis=-1)
    return float(mx.mean(per_row.astype(mx.float32)))


def span_token_accuracy(predicted: mx.array, target: mx.array, span: mx.array) -> float:
    """Fraction of answer positions that match."""
    agree = mx.logical_and(mx.equal(predicted, target), span)
    return float(mx.sum(agree.astype(mx.float32)) / mx.maximum(mx.sum(span.astype(mx.float32)), 1e-6))


def total_layer_applications(cfg: ModelConfig, loop_counts) -> int:
    """Transformer block evaluations for a whole sampling trajectory."""
    return int(sum(layer_applications(cfg, int(r)) for r in loop_counts))


def loop_profile_by_timestep(loop_counts, steps: int, buckets: int = 4) -> list:
    """Average loop count per timestep bucket, early -> late denoising.

    Bucketing is what makes the paper's Figure 6 claim ("more loops at
    intermediate timesteps") measurable on a short trajectory.
    """
    if len(loop_counts) != steps:
        raise ValueError("loop_counts must have one entry per denoising step")
    size = max(steps // buckets, 1)
    profile = []
    for b in range(0, steps, size):
        chunk = loop_counts[b : b + size]
        profile.append(sum(chunk) / len(chunk))
    return profile
