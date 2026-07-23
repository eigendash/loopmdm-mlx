"""The synthetic corpus must actually contain the task it claims to."""

from __future__ import annotations

import numpy as np
import pytest

from loopmdm.data import (
    BOS_ID,
    EOS_ID,
    MASK_ID,
    PAD_ID,
    PARITY_MARK,
    STOI,
    TOKENS,
    VOCAB_SIZE,
    decode,
    encode,
    generate_examples,
    task_counts,
    to_array,
)

SEQ = 16


def test_vocabulary_is_consistent():
    assert len(TOKENS) == VOCAB_SIZE == len(set(TOKENS))
    assert STOI["<pad>"] == PAD_ID == 0
    assert STOI["<bos>"] == BOS_ID == 1
    assert STOI["<eos>"] == EOS_ID == 2
    # the mask is not a real token, it is an extra embedding row
    assert MASK_ID == VOCAB_SIZE
    assert "<mask>" not in TOKENS
    for ch in "0123456789+=?P " + "abcdefghijklmnopqrstuvwxyz":
        assert STOI[ch] == encode(ch)[0]
    assert decode(encode("the cat")) == "the cat"
    assert decode([MASK_ID]) == "<mask>"


def test_sum_examples_have_the_correct_answer():
    examples = generate_examples(200, SEQ, seed=1, task="sum")
    for e in examples:
        assert e.task == "sum"
        assert e.answer_start > 0 and e.answer_end == e.answer_start + 1
        prefix = decode(e.tokens[1 : e.answer_start])
        assert prefix.endswith("=")
        addends = [int(x) for x in prefix[:-1].split("+")]
        assert 2 <= len(addends) <= 4
        assert all(0 <= a <= 9 for a in addends)
        assert decode(e.answer_ids) == str(sum(addends) % 10)


def test_parity_examples_have_the_correct_answer():
    examples = generate_examples(200, SEQ, seed=2, task="parity")
    for e in examples:
        assert e.answer_end == e.answer_start + 2
        prefix = decode(e.tokens[1 : e.answer_start])
        assert prefix.endswith("?")
        addends = [int(x) for x in prefix[:-1].split("+")]
        answer = decode(e.answer_ids)
        assert answer == f"{PARITY_MARK}{sum(addends) % 2}"
        odd = sum(a % 2 for a in addends)
        assert int(answer[1]) == odd % 2


def test_sequences_are_fixed_length_and_padded_on_the_right():
    examples = generate_examples(150, SEQ, seed=3)
    for e in examples:
        assert e.tokens.shape == (SEQ,)
        assert e.tokens.dtype == np.int32
        non_pad = np.flatnonzero(e.tokens != PAD_ID)
        assert len(non_pad) > 0
        assert non_pad[-1] == len(non_pad) - 1, "padding must be a suffix"
        assert e.tokens[non_pad[-1]] == EOS_ID
        if e.has_answer:
            assert e.answer_end <= non_pad[-1]


def test_text_examples_have_no_answer_span():
    examples = generate_examples(50, SEQ, seed=4, task="text")
    for e in examples:
        assert e.task == "text"
        assert not e.has_answer
        assert e.answer_mask().sum() == 0
        assert not (e.answer_ids > 0).any()
        body = decode(e.tokens[1 : int(np.flatnonzero(e.tokens == EOS_ID)[0])])
        assert body.islower()
        assert body.strip() == body and len(body) > 0


def test_answer_mask_selects_exactly_the_answer():
    examples = generate_examples(30, SEQ, seed=5, task="sum")
    for e in examples:
        mask = e.answer_mask()
        assert mask.sum() == 1
        assert np.array_equal(np.flatnonzero(mask), [e.answer_start])
        assert np.array_equal(e.tokens[mask], e.answer_ids)


def test_generation_is_seeded_and_different_seeds_differ():
    a = generate_examples(40, SEQ, seed=7)
    b = generate_examples(40, SEQ, seed=7)
    c = generate_examples(40, SEQ, seed=8)
    assert all(np.array_equal(x.tokens, y.tokens) for x, y in zip(a, b))
    assert any(not np.array_equal(x.tokens, y.tokens) for x, y in zip(a, c))


def test_task_mix_is_respected():
    examples = generate_examples(4000, SEQ, seed=9)
    counts = task_counts(examples)
    total = sum(counts.values())
    assert counts["sum"] / total == pytest.approx(0.5, abs=0.03)
    assert counts["parity"] / total == pytest.approx(0.25, abs=0.03)
    assert counts["text"] / total == pytest.approx(0.25, abs=0.03)


def test_addend_count_is_bounded():
    examples = generate_examples(300, SEQ, seed=10, task="sum", max_addends=3)
    for e in examples:
        prefix = decode(e.tokens[1 : e.answer_start])
        assert 2 <= len(prefix[:-1].split("+")) <= 3
    with pytest.raises(ValueError):
        generate_examples(2, SEQ, seed=0, task="nope")


def test_to_array_shapes_and_values():
    examples = generate_examples(12, SEQ, seed=11, task="sum")
    arr = to_array(examples)
    assert arr.shape == (12, SEQ)
    assert arr.dtype == np.int32 or str(arr.dtype) == "mlx.core.int32"
    assert int(arr.min()) >= 0
    assert int(arr.max()) < MASK_ID
    for row, e in zip(np.array(arr), examples):
        assert np.array_equal(row, e.tokens)
