# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare packed cache IDs/masks with the full row multiset, including duplicates."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import jax
import numpy as np
from haliax import Axis

from experiments.june_tpu_67b_a2b.moe import vista_snowball_chat as v


def fingerprint(ids: np.ndarray, loss: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(ids, dtype="<i4").tobytes())
    digest.update(np.asarray(loss, dtype="i1").tobytes())
    return digest.hexdigest()


def main() -> None:
    stage, rows_path, parquet_dir, cache, tokenizer = sys.argv[1:]
    if not (Path(parquet_dir) / "parquet.list").is_file():
        raise ValueError("Missing prepared parquet list")
    expected = Counter()
    row_tokens = trained = rows = 0
    with open(rows_path) as stream:
        for line in stream:
            row = json.loads(line)
            expected[fingerprint(np.asarray(row["ids"]), np.asarray(row["loss"]))] += 1
            row_tokens += len(row["ids"])
            trained += sum(row["loss"])
            rows += 1
    provenance = v.validate_cache_provenance(cache, stage, tokenizer)
    tokens, examples, shards = v.validate_chat_cache_layout(cache, expect_tokens=None, expect_examples=None)
    datasets = v.snowball_chat_data_config(cache_path=cache, tokenizer_path=tokenizer, stage=stage).train_sets(
        Axis("position", v.SNOWBALL_CHAT_SEQUENCE_LENGTH),
        key=jax.random.PRNGKey(v.SNOWBALL_CHAT_SEED),
        initial_batch_size=v.SNOWBALL_CHAT_BATCH_SIZE,
    )
    dataset = datasets[v.STAGES[stage].component].as_sync_dataset()
    observed = Counter()
    observed_trained = 0
    for index in range(len(dataset)):
        example = dataset[index]
        ids = np.asarray(example.tokens)
        weight = np.asarray(example.loss_weight)
        segments = np.asarray(example.attn_mask.segment_ids[0])
        if np.any(weight[segments < 0] != 0) or not np.all(np.isin(weight, [0, 1])):
            raise ValueError(f"Invalid mask or weighted padding in pack {index}")
        cursor = 0
        while cursor < len(ids):
            if segments[cursor] < 0:
                cursor += 1
                continue
            end = cursor + 1
            while end < len(ids) and segments[end] == segments[cursor]:
                end += 1
            if weight[end - 1] != 0:
                raise ValueError(f"Loss crosses a row boundary in pack {index}")
            loss = np.zeros(end - cursor, dtype=np.int8)
            loss[1:] = weight[cursor : end - 1]
            observed[fingerprint(ids[cursor:end], loss)] += 1
            observed_trained += int(weight[cursor:end].sum())
            cursor = end
    if expected != observed:
        raise ValueError(
            f"Packed IDs/masks differ: missing={sum((expected-observed).values())}, "
            f"extra={sum((observed-expected).values())}"
        )
    if tokens != row_tokens or observed_trained != trained:
        raise ValueError("Packed token totals differ from the prepared rows")
    print(
        json.dumps(
            {
                "verdict": "EQUAL",
                "stage": stage,
                "rows": rows,
                "distinct_trajectories": len(expected),
                "duplicate_rows": rows - len(expected),
                "tokens": tokens,
                "trained_tokens": trained,
                "cache_examples": examples,
                "cache_shards": shards,
                "packs": len(dataset),
                "format": provenance["format"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
