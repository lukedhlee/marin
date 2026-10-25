# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fractional student targets must survive the actual cache and causal packing path."""

import asyncio

import haliax as hax
import numpy as np
import pytest
from levanter.data.text.datasets import ChatDataset
from levanter.store.cache import SerialCacheWriter
from tokenizers import Tokenizer, models, pre_tokenizers

from experiments.june_tpu_67b_a2b.moe.grug_datakit_chat import PrerenderedRowProcessor
from experiments.june_tpu_67b_a2b.moe.student_target_labels import add_student_targets


@pytest.mark.parametrize("weight", [0.0, 0.1])
def test_student_weight_survives_cache_packing_and_next_token_shift(tmp_path, weight):
    processor = PrerenderedRowProcessor(
        ids_field="ids",
        loss_field="loss",
        max_tokens=8,
        vocab_size=128256,
        bos_id=128000,
        student_loss_weight=weight,
    )
    student = 2 if weight else 0
    rows = [
        {"id": "a", "ids": [128000, 10, 11, 12], "loss": [0, 0, student, 1]},
        {"id": "b", "ids": [128000, 20, 21], "loss": [0, student, 1]},
    ]
    processed = processor(rows)
    if weight == 0:
        for row, original in zip(processed, rows, strict=True):
            np.testing.assert_array_equal(row["assistant_masks"], np.asarray(original["loss"], np.int32))
            assert row["assistant_masks"].dtype == np.int32
    with SerialCacheWriter(str(tmp_path / "cache"), processor.output_exemplar) as writer:
        writer.write_batch(processed)
    dataset = ChatDataset(writer.result(), hax.Axis("position", 8), max_segments_per_example=4)
    example = asyncio.run(dataset.getitem_async(0))
    np.testing.assert_array_equal(np.asarray(example.tokens)[:7], [128000, 10, 11, 12, 128000, 20, 21])
    # Predict the student token at0.1 and teacher at1, never the next document's BOS or padding.
    expected = np.asarray([0, weight, 1, 0, weight, 1, 0, 0], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(example.loss_weight), expected)


@pytest.mark.parametrize("bad", [0.1, -1, 3, float("nan"), float("inf")])
def test_weighted_ownership_cache_rejects_invalid_labels_instead_of_rounding(bad):
    processor = PrerenderedRowProcessor(
        ids_field="ids",
        loss_field="loss",
        max_tokens=8,
        vocab_size=128256,
        bos_id=128000,
        student_loss_weight=0.1,
    )
    with pytest.raises(ValueError):
        processor([{"ids": [128000, 10, 11], "loss": [0, bad, 1]}])


@pytest.mark.parametrize("length", [9, 13])
def test_student_owner_labels_exclude_headers_and_preserve_teacher_targets(length):
    text = "BOS USER question STUDENT wrong bad END TEACHER fixed END ROUTER stop END"
    vocab = {word: i for i, word in enumerate(dict.fromkeys(text.split()))}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    ids = tokenizer.encode(text).ids
    mask = [0] * len(ids)
    mask[8:10] = [1, 1]
    source = {
        "sid": "example",
        "ids": ids,
        "loss": mask,
        "turns": [
            {"owner": "student", "span": [text.index("wrong"), text.index(" TEACHER")]},
            {"owner": "teacher", "span": [text.index("fixed"), text.index(" ROUTER")]},
            {"owner": "router", "span": [text.index("stop"), len(text)]},
        ],
    }
    row = {**source, "ids": ids[:length], "loss": mask[:length]}
    result, stats = add_student_targets(row, source, tokenizer)
    expected = list(mask)
    expected[4:7] = [2, 2, 2]
    assert result["ids"] == row["ids"]
    assert result["loss"] == expected[:length]
    assert stats["student_tokens"] == 3
