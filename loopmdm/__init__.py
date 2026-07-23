"""Toy reimplementation of selective layer looping for masked diffusion LMs.

The public surface is small on purpose: a ``ModelConfig`` describing the
head/mid/tail split, a functional parameter tree with a ``forward`` that takes
the loop count as an argument, and the schedules/samplers around it.
"""

from .config import ModelConfig, TrainConfig
from .data import (
    MASK_ID,
    VOCAB_SIZE,
    Example,
    decode,
    encode,
    generate_examples,
    to_array,
)
from .metrics import (
    answer_nll,
    mask_to_mask_attention,
    mid_attention_by_iteration,
    span_exact_match,
    span_token_accuracy,
    total_layer_applications,
)
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
from .sampler import (
    SampleTrace,
    confidence_sample,
    masked_prompt,
    stochastic_sample,
)
from .schedule import Schedule, corrupt_tokens, sample_reverse_step
from .train import TrainResult, mdm_loss, train_mdm

__all__ = [
    "Example",
    "MASK_ID",
    "ModelConfig",
    "VOCAB_SIZE",
    "TrainConfig",
    "ForwardOutput",
    "SampleTrace",
    "Schedule",
    "TrainResult",
    "confidence_sample",
    "corrupt_tokens",
    "decode",
    "encode",
    "generate_examples",
    "answer_nll",
    "mask_to_mask_attention",
    "masked_prompt",
    "mid_attention_by_iteration",
    "mdm_loss",
    "count_params",
    "forward",
    "forward_adaptive",
    "init_params",
    "layer_applications",
    "num_parameters",
    "relative_hidden_change",
    "sample_reverse_step",
    "span_exact_match",
    "span_token_accuracy",
    "stochastic_sample",
    "total_layer_applications",
    "to_array",
    "train_mdm",
    "unroll_params",
    "unrolled_config",
]
