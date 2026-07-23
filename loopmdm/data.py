"""A tiny synthetic corpus with an arithmetic task in it.

The task is designed so that a masked diffusion model has to *use the masked
positions together* rather than copy from nearby: given ``a1+a2+...`` the model
must produce the two-digit sum (carries couple the digit positions), and given
the same expression followed by ``?`` it must produce ``p0``/``p1`` for the
parity, which depends on every addend at once.  A third, smaller slice of the
corpus is short templated sentences so the model also sees ordinary text
statistics.

Every example is a fixed-length integer sequence padded with ``<pad>``; the
answer span is recorded so that evaluation can look only at the answer tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

SPECIALS = ["<pad>", "<bos>", "<eos>"]
DIGITS = list("0123456789")
#: "P" is the parity-answer marker; it is upper case so that it cannot
#: collide with the letter "p" used by the text templates
SYMBOLS = ["+", "=", "?", "P"]
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
SPACE = " "

#: real output tokens; the absorbing mask is an *extra* input embedding row
TOKENS = SPECIALS + DIGITS + SYMBOLS + LETTERS + [SPACE]
STOI = {tok: i for i, tok in enumerate(TOKENS)}
VOCAB_SIZE = len(TOKENS)
MASK_ID = VOCAB_SIZE
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2

PARITY_MARK = "P"

#: short templated sentences; they must fit in a :class:`ModelConfig` sequence
TEXT_SENTENCES = [
    "the cat",
    "a dog",
    "the sun",
    "old tree",
    "birds",
    "he ran",
    "a map",
    "the sea",
    "cold river",
    "she wrote",
]

TASK_MIX = {"sum": 0.5, "parity": 0.25, "text": 0.25}


def encode(text: str) -> list[int]:
    return [STOI[ch] for ch in text]


def decode(ids) -> str:
    out = []
    for i in ids:
        i = int(i)
        if i == MASK_ID:
            out.append("<mask>")
        elif 0 <= i < VOCAB_SIZE:
            out.append(TOKENS[i])
        else:
            out.append("?")
    return "".join(out)


@dataclass
class Example:
    """One corpus item: fixed-length tokens plus the answer span (if any)."""

    tokens: np.ndarray  # (seq_len,) int32, padded with PAD_ID on the right
    task: str
    answer_start: int  # -1 when the example has no answer (plain text)
    answer_end: int  # exclusive

    @property
    def has_answer(self) -> bool:
        return self.answer_start >= 0

    @property
    def answer_ids(self) -> np.ndarray:
        return self.tokens[self.answer_start : self.answer_end]

    def answer_mask(self) -> np.ndarray:
        mask = np.zeros_like(self.tokens, dtype=bool)
        if self.has_answer:
            mask[self.answer_start : self.answer_end] = True
        return mask


def _arithmetic_text(rng: np.random.Generator, k: int, parity: bool) -> tuple[str, int, int]:
    """Build the string and return it with the (start, length) of the answer.

    The sum answer is a single digit, ``(a1 + ... + ak) mod 10``: the interesting
    part is aggregating every addend, and one token keeps the answer span small
    enough for the toy model to learn it in the experiment's budget.  The parity
    answer is ``P`` followed by the parity bit.
    """
    addends = rng.integers(0, 10, size=k)
    total = int(addends.sum())
    prefix = "+".join(str(int(a)) for a in addends)
    if parity:
        text = f"{prefix}?{PARITY_MARK}{total % 2}"
        return text, len(text) - 2, 2
    text = f"{prefix}={total % 10}"
    return text, len(text) - 1, 1


def make_example(
    rng: np.random.Generator, seq_len: int, task: str, max_addends: int = 4
) -> Example:
    """One example of the requested task, padded/truncated to ``seq_len``."""
    width = 0
    if task == "text":
        text = str(rng.choice(TEXT_SENTENCES))
        ids = [BOS_ID] + encode(text) + [EOS_ID]
        start = -1
    elif task in ("sum", "parity"):
        k = int(rng.integers(2, max_addends + 1))
        text, offset, width = _arithmetic_text(rng, k, parity=(task == "parity"))
        ids = [BOS_ID] + encode(text) + [EOS_ID]
        if len(ids) > seq_len:
            raise ValueError(
                f"seq_len {seq_len} is too short for a {k}-addend {task} example"
            )
        start = 1 + offset
    else:
        raise ValueError(f"unknown task {task!r}")

    end = start + width if start >= 0 else -1
    if len(ids) > seq_len:
        # only the templated sentences can be too long: keep <eos> at the end
        ids = ids[: seq_len - 1] + [EOS_ID]
    ids = ids + [PAD_ID] * (seq_len - len(ids))
    return Example(tokens=np.array(ids, dtype=np.int32), task=task, answer_start=start, answer_end=end)


def generate_examples(
    n: int,
    seq_len: int,
    seed: int = 0,
    mix: dict | None = None,
    max_addends: int = 4,
    task: str | None = None,
) -> list[Example]:
    """Generate ``n`` examples; deterministic given ``seed``.

    ``task`` forces a single task, otherwise the ``mix`` proportions are used.
    """
    rng = np.random.default_rng(seed)
    mix = mix or TASK_MIX
    names = list(mix)
    probs = np.array([mix[name] for name in names], dtype=np.float64)
    probs = probs / probs.sum()
    if task is not None:
        names, probs = [task], np.array([1.0])
    chosen = rng.choice(len(names), size=n, p=probs)
    return [make_example(rng, seq_len, names[int(c)], max_addends) for c in chosen]


def to_array(examples: list[Example]) -> mx.array:
    return mx.array(np.stack([e.tokens for e in examples]))


def task_counts(examples: list[Example]) -> dict:
    counts: dict = {}
    for e in examples:
        counts[e.task] = counts.get(e.task, 0) + 1
    return counts
