# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rendered relay rows to the one-shard Parquet a cache is built from, and the steps in one pass over its packs.

OpenThoughts-Agent renders each episode to a JSON line ``{sid, ids, loss, n_tokens, fits, turns}`` with the base's
tokenizer. The Parquet keeps ``id`` (the episode's session id), ``ids`` (int32), ``loss`` (int8) and ``n_tokens``,
in the input order, which is also the cache's order for a one-shard cache.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.data.packing import pack_documents

from experiments.post_training.relay_sft.recipe import MAX_SEGMENTS_PER_PACK

PARQUET_NAME = "train-00000-of-00001.parquet"
_REFUSALS_SHOWN = 20


@dataclass(frozen=True)
class PackCount:
    rows: int
    tokens: int
    packs: int
    epoch_steps: int
    """Steps in one pass over every pack: ceil(packs / batch size)."""


def row_problem(row: dict, *, max_tokens: int, bos_id: int, vocab_size: int) -> str | None:
    """Why the cache would refuse or silently change this row, or None if it is fine."""
    ids, loss = row["ids"], row["loss"]
    if len(ids) != len(loss) or len(ids) != row.get("n_tokens", len(ids)):
        return "ids, loss and n_tokens disagree in length"
    if row.get("fits") is False or len(ids) > max_tokens:
        return f"{len(ids)} tokens > {max_tokens}"
    if any(value not in (0, 1) for value in loss):
        return "loss values other than 0 and 1"
    if ids[0] != bos_id or loss[0] != 0:
        return "no untrained BOS first"
    if not any(loss):
        return "no trained token"
    if min(ids) < 0 or max(ids) >= vocab_size:
        return "id outside the vocabulary"
    return None


def rows_table(rows_path: Path, *, max_tokens: int, bos_id: int, vocab_size: int) -> pa.Table:
    """Read and check every row; any refused row (or a repeated session id) fails the whole file."""
    sids: list[str] = []
    ids: list[list[int]] = []
    losses: list[list[int]] = []
    refused: list[dict] = []
    seen: set[str] = set()
    with rows_path.open() as handle:
        for line_number, line in enumerate(handle):
            row = json.loads(line)
            problem = row_problem(row, max_tokens=max_tokens, bos_id=bos_id, vocab_size=vocab_size)
            if problem is None and row["sid"] in seen:
                problem = "repeated session id"
            if problem is not None:
                refused.append({"line": line_number, "sid": row["sid"], "problem": problem})
                continue
            seen.add(row["sid"])
            sids.append(row["sid"])
            ids.append(row["ids"])
            losses.append(row["loss"])
    if refused:
        raise ValueError(f"{len(refused)} rows refused in {rows_path}; first: {refused[:_REFUSALS_SHOWN]}")
    return pa.table(
        {
            "id": pa.array(sids, pa.string()),
            "ids": pa.array(ids, pa.list_(pa.int32())),
            "loss": pa.array(losses, pa.list_(pa.int8())),
            "n_tokens": pa.array([len(row_ids) for row_ids in ids], pa.int32()),
        }
    )


def write_rows_parquet(table: pa.Table, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / PARQUET_NAME
    pq.write_table(table, shard, row_group_size=64)
    return shard


def count_packs(parquet_path: Path, *, sequence_length: int, batch_size: int) -> PackCount:
    """Pack the rows with Levanter's packer, in shard order, as the trainer's chat dataset will."""
    lengths = pq.read_table(parquet_path, columns=["n_tokens"])["n_tokens"].to_numpy().astype(np.int64)
    packs = len(pack_documents(lengths, sequence_length, MAX_SEGMENTS_PER_PACK, slice_strategy="left"))
    return PackCount(
        rows=len(lengths), tokens=int(lengths.sum()), packs=packs, epoch_steps=math.ceil(packs / batch_size)
    )
