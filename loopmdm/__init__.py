"""Toy reimplementation of selective layer looping for masked diffusion LMs.

The public surface is small on purpose: a ``ModelConfig`` describing the
head/mid/tail split, a functional parameter tree with a ``forward`` that takes
the loop count as an argument, and the schedules/samplers around it.
"""

from .config import ModelConfig, TrainConfig
from .model import (
    ForwardOutput,
    count_params,
    forward,
    forward_adaptive,
    init_params,
    layer_applications,
    num_parameters,
    relative_hidden_change,
    unroll_params,
    unrolled_config,
)
from .schedule import Schedule, corrupt_tokens, sample_reverse_step
from .train import TrainResult, mdm_loss, train_mdm

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "ForwardOutput",
    "Schedule",
    "TrainResult",
    "corrupt_tokens",
    "mdm_loss",
    "count_params",
    "forward",
    "forward_adaptive",
    "init_params",
    "layer_applications",
    "num_parameters",
    "relative_hidden_change",
    "sample_reverse_step",
    "train_mdm",
    "unroll_params",
    "unrolled_config",
]
