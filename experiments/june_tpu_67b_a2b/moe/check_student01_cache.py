# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify actual packed training examples contain fractional student targets."""

import asyncio
import json
import sys
from pathlib import Path

import haliax as hax
import numpy as np
from levanter.data.text.datasets import ChatDataset
from levanter.store.cache import TreeCache


def main() -> None:
    cache_path, root = sys.argv[1:]
    exemplar = {"input_ids": np.zeros((0,), np.int32), "assistant_masks": np.zeros((0,), np.float32)}
    cache = TreeCache.load(cache_path + "/train", exemplar)
    dataset = ChatDataset(cache, hax.Axis("position", 65536), max_segments_per_example=64)
    counts = {"student": 0, "teacher": 0, "context": 0}
    for i in range(8):
        example = asyncio.run(dataset.getitem_async(i))
        weights = np.asarray(example.loss_weight)
        assert weights.dtype == np.float32 and np.isfinite(weights).all()
        assert np.isin(weights, np.asarray([0, 0.1, 1], np.float32)).all()
        counts["student"] += int(np.count_nonzero(weights == np.float32(0.1)))
        counts["teacher"] += int(np.count_nonzero(weights == 1))
        counts["context"] += int(np.count_nonzero(weights == 0))
    assert counts["student"] > 0 and counts["teacher"] > 0 and counts["context"] > 0, counts
    Path(root, "packed_mask_gate.json").write_text(json.dumps(counts, indent=2) + "\n")
    print("PACKED_STUDENT01_MASK_OK", json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
