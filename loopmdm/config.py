"""Configuration for the toy looped masked diffusion model.

The denoising network is split into three sequential groups, following the
paper's notation: a *head* of ``head_layers`` layers, a *mid-block* of
``mid_layers`` layers whose weights are shared across loop iterations, and a
*tail* of ``tail_layers`` layers plus the output projection.  Only the
mid-block is looped; the head and the tail are applied exactly once per
forward pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the denoising transformer.

    ``head_layers + mid_layers + tail_layers == n_layers`` and the number of
    parameters does not depend on how many times the mid-block is applied.
    """

    vocab_size: int
    seq_len: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 6
    head_layers: int = 1
    mid_layers: int = 2
    mlp_ratio: float = 2.0
    time_dim: int = 64
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("RoPE requires an even per-head dimension")
        if self.head_layers < 0 or self.mid_layers < 1:
            raise ValueError("need a non-negative head and at least one mid layer")
        if self.head_layers + self.mid_layers > self.n_layers:
            raise ValueError(
                "head_layers + mid_layers exceeds n_layers "
                f"({self.head_layers} + {self.mid_layers} > {self.n_layers})"
            )
        if self.seq_len < 1 or self.vocab_size < 2:
            raise ValueError("seq_len and vocab_size must be positive")

    @property
    def tail_layers(self) -> int:
        return self.n_layers - self.head_layers - self.mid_layers

    @property
    def mask_id(self) -> int:
        """Id of the absorbing mask token.

        The mask is not a real token: the embedding table has ``vocab_size + 1``
        rows (the extra row is the mask) while the LM head predicts ``vocab_size``
        real tokens only.
        """
        return self.vocab_size

    @property
    def padded_vocab(self) -> int:
        return self.vocab_size + 1

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mlp_hidden(self) -> int:
        return int(round(self.mlp_ratio * self.d_model))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings for the toy pretraining runs."""

    steps: int = 1000
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 0.01
    warmup: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    # Maximum loop count used during training.  The loop count is sampled
    # uniformly from {1, ..., max_loops} at every step (paper Eq. 5).
    max_loops: int = 1
    # "uniform" is the plain masked-token cross-entropy (weight 1); "nelbo"
    # applies the exact continuous-time weight -alpha'(t) / (1 - alpha(t)).
    loss_weighting: str = "uniform"
    t_min: float = 0.02
    # Upper bound of the sampled mask rate.  <1 keeps the sequence mostly
    # readable, which is what makes the toy arithmetic task learnable here.
    t_max: float = 1.0

    def __post_init__(self) -> None:
        if self.max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if self.loss_weighting not in ("uniform", "nelbo"):
            raise ValueError(f"unknown loss_weighting {self.loss_weighting!r}")
        if not 0.0 <= self.t_max <= 1.0:
            raise ValueError("t_max must lie in [0, 1]")

    def as_dict(self) -> dict:
        return asdict(self)
