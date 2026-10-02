# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.post_training.relay_sft.prerendered import PrerenderedRowProcessor

BOS = 7


def _processor() -> PrerenderedRowProcessor:
    return PrerenderedRowProcessor(ids_field="ids", loss_field="loss", max_tokens=6, vocab_size=10, bos_id=BOS)


def test_rows_pass_through_as_ids_and_assistant_mask():
    (row,) = _processor()([{"id": "a", "ids": [BOS, 3, 4, 5], "loss": [0, 0, 1, 1]}])

    np.testing.assert_array_equal(row["input_ids"], [BOS, 3, 4, 5])
    np.testing.assert_array_equal(row["assistant_masks"], [0, 0, 1, 1])
    assert row["input_ids"].dtype == np.int32 and row["assistant_masks"].dtype == np.int32


@pytest.mark.parametrize(
    ("ids", "loss", "problem"),
    [
        ([BOS, 3, 4], [0, 1], "differ"),
        ([BOS, 1, 2, 3, 4, 5, 6], [0, 1, 1, 1, 1, 1, 1], "rows must hold"),
        ([BOS, 3, 12], [0, 1, 1], "outside"),
        ([BOS, 3, 4], [0, 2, 1], "other than 0 and 1"),
        ([3, BOS, 4], [0, 1, 1], "untrained BOS"),
        ([BOS, 3, 4], [1, 1, 1], "untrained BOS"),
        ([BOS, 3, 4], [0, 0, 0], "trains no token"),
    ],
)
def test_rows_the_cache_would_corrupt_are_refused(ids, loss, problem):
    with pytest.raises(ValueError, match=problem):
        _processor()([{"id": "bad", "ids": ids, "loss": loss}])
