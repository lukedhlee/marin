# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow.parquet as pq
import pytest

from experiments.post_training.relay_sft.rows import count_packs, rows_table, write_rows_parquet

BOS = 1


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(sid, n_tokens):
    return {"sid": sid, "ids": [BOS] + [2] * (n_tokens - 1), "loss": [0] + [1] * (n_tokens - 1), "n_tokens": n_tokens}


def test_rows_round_trip_through_parquet_and_count_packs(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    _write_rows(rows_path, [_row("a", 6), _row("b", 4), _row("c", 4), _row("d", 9)])

    table = rows_table(rows_path, max_tokens=10, bos_id=BOS, vocab_size=8)
    shard = write_rows_parquet(table, tmp_path / "out")
    written = pq.read_table(shard)

    assert written["id"].to_pylist() == ["a", "b", "c", "d"]
    assert written["n_tokens"].to_pylist() == [6, 4, 4, 9]
    assert written["loss"].to_pylist()[1] == [0, 1, 1, 1]
    # greedy, in order, at most 10 tokens per pack: [6 4] [4] [9]
    packs = count_packs(shard, sequence_length=10, batch_size=2)
    assert (packs.rows, packs.tokens, packs.packs, packs.epoch_steps) == (4, 23, 3, 2)


@pytest.mark.parametrize(
    ("rows", "problem"),
    [
        ([_row("a", 4), _row("a", 5)], "repeated session id"),
        ([_row("a", 11)], "11 tokens > 10"),
        ([{**_row("a", 4), "loss": [0, 0, 0, 0]}], "no trained token"),
    ],
)
def test_any_refused_row_fails_the_file(tmp_path, rows, problem):
    rows_path = tmp_path / "rows.jsonl"
    _write_rows(rows_path, rows)

    with pytest.raises(ValueError, match=problem):
        rows_table(rows_path, max_tokens=10, bos_id=BOS, vocab_size=8)
