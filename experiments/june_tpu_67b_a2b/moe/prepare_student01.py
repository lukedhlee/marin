# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep every baseline token and teacher target; add label2 only inside recorded student turns."""

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from experiments.june_tpu_67b_a2b.moe.student_target_labels import add_student_targets


def main() -> None:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    root = Path(os.environ["FP32_REPLAY_ROOT"])
    scratch = Path(f"/scratch/11584/{os.environ['USER']}")
    baseline = scratch / "experiments/relay-kimi-fp32-capacity2-20261021"
    reference = json.loads((baseline / "prepared.json").read_text())
    selected = scratch / "experiments/relay-fixed-design-20261021/final/fdr/fdr_selected.jsonl"
    cf_rows = Path(reference["sources"]["calibforge"]["path"])
    kimi = Path(reference["sources"]["kimi"]["path"])
    assert not (root / "rows.jsonl").exists(), "Refusing to overwrite prepared or partial rows"
    tokenizer = Tokenizer.from_file(str(scratch / "models/grug-datakit-sft-20260921/tokenizer.json"))
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    shard = data / "train-00000-of-00001.parquet"
    schema = pa.schema(
        [("id", pa.string()), ("ids", pa.list_(pa.int32())), ("loss", pa.list_(pa.int8())), ("n_tokens", pa.int32())]
    )
    stats = Counter()
    seen = set()
    buffer = []
    source_digest = hashlib.sha256()
    kimi_digest = hashlib.sha256()
    with pq.ParquetWriter(str(shard), schema) as writer, (root / "rows.jsonl").open("xb") as output:

        def write_row(row, line):
            assert row["sid"] not in seen
            seen.add(row["sid"])
            assert row["fits"] and 0 < row["n_tokens"] <= 65536
            assert len(row["ids"]) == len(row["loss"]) == row["n_tokens"]
            assert row["ids"][0] == 128000 and row["loss"][0] == 0
            assert set(row["loss"]) <= {0, 1, 2} and any(row["loss"])
            assert min(row["ids"]) >= 0 and max(row["ids"]) < 128256
            output.write(line)
            buffer.append({"id": row["sid"], "ids": row["ids"], "loss": row["loss"], "n_tokens": row["n_tokens"]})
            stats["rows"] += 1
            stats["tokens"] += row["n_tokens"]
            stats["teacher_tokens"] += row["loss"].count(1)
            if len(buffer) == 64:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()

        with cf_rows.open("rb") as rows, selected.open() as selections:
            for line, selection in zip(rows, selections, strict=True):
                source_digest.update(line)
                row = json.loads(line)
                sel = json.loads(selection)
                assert row["sid"] == sel["sid"]
                with open(sel["source_path"], "rb") as stream:
                    stream.seek(sel["source_offset"])
                    source = json.loads(stream.readline())
                assert source["sid"] == row["sid"]
                labeled, counts = add_student_targets(row, source, tokenizer)
                stats.update(counts)
                stats["calibforge_rows"] += 1
                write_row(labeled, (json.dumps(labeled) + "\n").encode())
        assert source_digest.hexdigest() == reference["sources"]["calibforge"]["sha256"]
        with kimi.open("rb") as stream:
            for line in stream:
                kimi_digest.update(line)
                row = json.loads(line)
                assert set(row["loss"]) <= {0, 1}
                stats["kimi_rows"] += 1
                write_row(row, line)
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    assert kimi_digest.hexdigest() == reference["sources"]["kimi"]["selected_sha256"]
    assert stats["calibforge_rows"] == 2974 and stats["kimi_rows"] == 4392 and stats["student_tokens"] > 0
    assert stats["teacher_tokens"] == sum(x["trained_tokens"] for x in reference["counts"].values())
    (data / "parquet.list").write_text(str(shard) + "\n")
    ota = Path(os.environ["REPLAY_OTA"])
    packing = subprocess.check_output(
        [
            sys.executable,
            str(ota / "data/relay/sft/pack_epoch_steps.py"),
            str(shard),
            "--seq-len",
            "65536",
            "--batch",
            "16",
        ],
        text=True,
    )
    packed = json.loads(packing.splitlines()[-1])
    assert packed == reference["packing"], (packed, reference["packing"])
    assert packed["pack_epoch_steps"] == 241
    (root / "pack_epoch_steps.json").write_text(json.dumps(packed) + "\n")
    report = {
        "baseline": str(baseline),
        "student_loss_weight": 0.1,
        "counts": dict(stats),
        "packing": packed,
        "weighted_target_mass": stats["teacher_tokens"] + 0.1 * stats["student_tokens"],
        "source_sha256": source_digest.hexdigest(),
        "kimi_sha256": kimi_digest.hexdigest(),
        "marin_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (root / "prepared.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PRIORITY_DATA_OK", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
