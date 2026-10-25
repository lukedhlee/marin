# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add student ownership labels using full-source token offsets, never a truncated decode."""

import bisect
from itertools import pairwise


def add_student_targets(row: dict, source: dict, tokenizer) -> tuple[dict, dict]:
    n = len(row["ids"])
    assert row["ids"] == source["ids"][:n], (row["sid"], "source token prefix changed")
    assert row["loss"] == source["loss"][:n], (row["sid"], "teacher mask changed")
    text = tokenizer.decode(source["ids"], skip_special_tokens=False)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    assert encoded.ids == source["ids"], (row["sid"], "full source does not roundtrip")
    turns = sorted(source["turns"], key=lambda turn: turn["span"][0])
    assert all(0 <= t["span"][0] < t["span"][1] <= len(text) for t in turns)
    assert all(a["span"][1] <= b["span"][0] for a, b in pairwise(turns))
    starts = [t["span"][0] for t in turns]
    labels = list(row["loss"])
    students = repaired = 0
    for i, (start, end) in enumerate(encoded.offsets[:n]):
        j = bisect.bisect_right(starts, start) - 1
        if j < 0 or end <= start or end > turns[j]["span"][1]:
            continue
        turn = turns[j]
        if turn["owner"] == "student":
            assert labels[i] == 0, (row["sid"], "baseline trains a student token")
            labels[i] = 2
            students += 1
            repaired += int(bool(turn.get("autofix")))
    assert all(new == old or (old == 0 and new == 2) for old, new in zip(row["loss"], labels, strict=True))
    return {**row, "loss": labels}, {"student_tokens": students, "autofixed_student_tokens": repaired}
