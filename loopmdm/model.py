"""The looped masked diffusion denoising network.

Layout (paper Section 3): the transformer is split into a ``head`` of
``head_layers`` layers, a ``mid`` block of ``mid_layers`` layers, and a ``tail``
of ``tail_layers`` layers plus the output projection.  The head and tail are
applied exactly once; the mid block is applied ``R`` (paper: ``S``) times with
the *same* weights::

    H^(0) = head(x_t),   H^(k) = mid(H^(k-1)),   p(x_0 | x_t) = tail(H^(R))

so the forward pass has an effective depth of ``head + R * mid + tail`` layers
while the parameter count stays at ``head + mid + tail``.

Parameters are a plain pytree of ``mlx.core`` arrays, not an ``nn.Module``.
Sharing is therefore literal array sharing: the loop reads the same arrays R
times inside one traced function, and ``mx.grad`` accumulates the cotangents of
every iteration into them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.utils as mu

from .config import ModelConfig
from .layers import (
    block_forward,
    init_layer_params,
    init_time_params,
    rms_norm,
    rope_tables,
    time_mlp,
)

__all__ = [
    "ForwardOutput",
    "init_params",
    "forward",
    "forward_adaptive",
    "unroll_params",
    "count_params",
    "layer_applications",
    "num_parameters",
]


@dataclass
class ForwardOutput:
    """Result of one denoiser forward pass."""

    logits: mx.array
    #: attention probabilities of every block evaluation, when ``capture=True``
    attentions: list | None = None
    #: hidden state after each mid-block application, when ``return_states=True``
    loop_states: list | None = None
    #: number of mid-block applications actually used (adaptive forward only)
    loops_used: int | None = None
    #: relative hidden-state change of each loop iteration (adaptive forward only)
    rel_changes: list | None = None

    @property
    def n_blocks(self) -> int:
        return 0 if self.attentions is None else len(self.attentions)


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
def init_params(cfg: ModelConfig, key: mx.array) -> dict:
    """Initialise the full parameter pytree.

    There is no learned positional embedding: positions enter through RoPE
    inside attention.  The embedding table has ``vocab_size + 1`` rows because
    the absorbing mask needs an input embedding; the LM head predicts the
    ``vocab_size`` real tokens only.
    """
    keys = mx.random.split(key, 7)
    d = cfg.d_model

    def layers(prefix: mx.array, n: int) -> list:
        if n == 0:
            return []
        return [init_layer_params(k, cfg) for k in mx.random.split(prefix, n)]

    return {
        "embed": mx.random.normal((cfg.padded_vocab, d), key=keys[0]) * 0.02,
        "time": init_time_params(keys[1], cfg),
        "head": layers(keys[2], cfg.head_layers),
        "mid": layers(keys[3], cfg.mid_layers),
        "tail": layers(keys[4], cfg.tail_layers),
        "out_norm": mx.ones((d,)),
        "out": mx.random.normal((cfg.vocab_size, d), key=keys[5]) * 0.02,
    }


def count_params(params: dict) -> int:
    """Number of *distinct* scalar parameters in a pytree.

    Arrays that appear more than once (the tied mid-block of an unrolled model)
    are counted once, which is what "looping adds no parameters" means.
    """
    leaves = mu.tree_flatten(params)
    seen: dict[int, int] = {}
    for _, array in leaves:
        if id(array) not in seen:
            seen[id(array)] = array.size
    return int(sum(seen.values()))


def num_parameters(cfg: ModelConfig) -> int:
    """Parameter count of the architecture, independent of the loop count."""
    return count_params(init_params(cfg, mx.random.key(0)))


def unroll_params(params: dict, loop_count: int) -> dict:
    """Build the equivalent *deeper* model: the tied mid-block repeated.

    The repeated entries are the same arrays, not copies, so the unrolled model
    is a weight-tied deep transformer and ``count_params`` is unchanged.
    """
    if loop_count < 1:
        raise ValueError("loop_count must be >= 1")
    out = dict(params)
    out["mid"] = [layer for _ in range(loop_count) for layer in params["mid"]]
    return out


def unrolled_config(cfg: ModelConfig, loop_count: int) -> ModelConfig:
    """Config of the deeper model whose depth matches ``R`` loop applications."""
    return replace(cfg, n_layers=cfg.n_layers + (loop_count - 1) * cfg.mid_layers)


def layer_applications(cfg: ModelConfig, loop_count: int) -> int:
    """Number of transformer blocks evaluated in one forward pass."""
    if loop_count < 1:
        raise ValueError("loop_count must be >= 1")
    return cfg.head_layers + loop_count * cfg.mid_layers + cfg.tail_layers


# --------------------------------------------------------------------------- #
# forward pass
# --------------------------------------------------------------------------- #
def _prepare(cfg: ModelConfig, tokens: mx.array, t, batch: int):
    cos, sin = rope_tables(tokens.shape[1], cfg.d_head, cfg.rope_base)
    t = mx.array(t, dtype=mx.float32) if not isinstance(t, mx.array) else t
    if t.ndim == 0:
        t = mx.broadcast_to(t.reshape(1), (batch,))
    elif t.shape[0] == 1 and batch > 1:
        t = mx.broadcast_to(t, (batch,))
    return cos, sin, t


def forward(
    params: dict,
    tokens: mx.array,
    t,
    cfg: ModelConfig,
    loop_count: int = 1,
    capture: bool = False,
    return_states: bool = False,
) -> ForwardOutput:
    """One denoiser forward pass with a fixed loop count."""
    if loop_count < 1:
        raise ValueError("loop_count must be >= 1")
    batch = tokens.shape[0]
    cos, sin, t = _prepare(cfg, tokens, t, batch)
    t_emb = time_mlp(params["time"], t, cfg)
    attentions: list | None = [] if capture else None
    states = []

    h = params["embed"][tokens]
    for layer in params["head"]:
        h, probs = block_forward(layer, h, t_emb, cos, sin, cfg, capture)
        if attentions is not None:
            attentions.append(probs)
    for _ in range(loop_count):
        for layer in params["mid"]:
            h, probs = block_forward(layer, h, t_emb, cos, sin, cfg, capture)
            if attentions is not None:
                attentions.append(probs)
        if return_states:
            states.append(h)
    for layer in params["tail"]:
        h, probs = block_forward(layer, h, t_emb, cos, sin, cfg, capture)
        if attentions is not None:
            attentions.append(probs)

    h = rms_norm(h, params["out_norm"])
    logits = h @ params["out"].T
    return ForwardOutput(
        logits=logits,
        attentions=attentions,
        loop_states=(states if return_states else None),
    )


def relative_hidden_change(previous: mx.array, current: mx.array) -> mx.array:
    """``||H^(k) - H^(k-1)||_F / ||H^(k)||_F``, the paper's adaptive stopping signal.

    The paper writes ``||H^(k) - H^(k-1)|| / ||H^(k)||`` without saying which
    norm; this uses the Frobenius norm over the whole hidden state (an
    interpretation, see the README).
    """
    num = mx.sqrt(mx.sum(mx.square(current - previous)))
    den = mx.sqrt(mx.sum(mx.square(current)))
    return num / mx.maximum(den, 1e-12)


def forward_adaptive(
    params: dict,
    tokens: mx.array,
    t,
    cfg: ModelConfig,
    max_loops: int,
    eps: float = 0.1,
    capture: bool = False,
    return_states: bool = False,
) -> ForwardOutput:
    """Forward pass that stops looping once the hidden state stabilises.

    The mid-block is applied at least once and at most ``max_loops`` times; after
    each application the relative hidden-state change is compared with ``eps``
    and looping stops as soon as it falls below the threshold (paper Section 4.3,
    ``eps = 0.1`` in their experiments).
    """
    if max_loops < 1:
        raise ValueError("max_loops must be >= 1")
    batch = tokens.shape[0]
    cos, sin, t = _prepare(cfg, tokens, t, batch)
    t_emb = time_mlp(params["time"], t, cfg)
    attentions: list | None = [] if capture else None
    states = []

    h = params["embed"][tokens]
    for layer in params["head"]:
        h, probs = block_forward(layer, h, t_emb, cos, sin, cfg, capture)
        if attentions is not None:
            attentions.append(probs)

    changes: list[mx.array] = []
    used = 0
    for _ in range(max_loops):
        previous = h
        for layer in params["mid"]:
            h, probs = block_forward(layer, h, t_emb, cos, sin, cfg, capture)
            if attentions is not None:
                attentions.append(probs)
        used += 1
        changes.append(relative_hidden_change(previous, h))
        if return_states:
            states.append(h)
        if float(changes[-1]) < eps:
            break

    for layer in params["tail"]:
        h, probs = block_forward(layer, h, t_emb, cos, sin, cfg, capture)
        if attentions is not None:
            attentions.append(probs)

    h = rms_norm(h, params["out_norm"])
    logits = h @ params["out"].T
    return ForwardOutput(
        logits=logits,
        attentions=attentions,
        loop_states=(states if return_states else None),
        loops_used=used,
        rel_changes=[float(c) for c in changes],
    )
