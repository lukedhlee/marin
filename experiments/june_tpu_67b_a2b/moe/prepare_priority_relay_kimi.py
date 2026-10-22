# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Freeze the matched original-Relay rows plus vanilla's exact frozen Kimi rows."""

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path(os.environ["FP32_REPLAY_ROOT"])
    scratch = Path(f"/scratch/11584/{os.environ['USER']}")
    vanilla = scratch / "experiments/vanilla-kimi-fp32-capacity2-20261021"
    reference = json.loads((vanilla / "prepared.json").read_text())
    expected_kimi = reference["sources"]["kimi"]["selected_sha256"]
    assert expected_kimi == "f76420a79f01f7ddba8a343f9f13dea7330bc9bb9e7b95c35ffefd3ade1276b1"
    fixed = scratch / "experiments/relay-fixed-design-20261021"
    task_counts = {}
    for arm in ("fdv", "fdr"):
        with (fixed / f"final/{arm}/{arm}_selected.jsonl").open() as stream:
            task_counts[arm] = Counter(json.loads(line)["task"] for line in stream)
    assert task_counts["fdv"] == task_counts["fdr"], "CalibForge task/trace counts differ from vanilla"
    assert len(task_counts["fdr"]) == 1487 and set(task_counts["fdr"].values()) == {2}
    with (fixed / "final/fdv/fdv_rows.jsonl").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == reference["sources"]["calibforge"]["sha256"]
    sources = {
        "calibforge": scratch / "experiments/relay-fixed-design-20261021/final/fdr/fdr_rows.jsonl",
        "kimi": vanilla / "kimi_rows.jsonl",
    }
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "prepared.json").exists(), "Preparation is already complete"
    rows = root / "rows.jsonl"
    assert not rows.exists(), "Refusing to replace partial or completed rows"
    counts = {}
    hashes = {}
    seen = set()
    with rows.open("xb") as output, (root / "kimi_rows.jsonl").open("xb") as kimi_output:
        for source, path in sources.items():
            before = path.stat()
            digest = hashlib.sha256()
            selected_digest = hashlib.sha256()
            count = tokens = trained = 0
            with path.open("rb") as stream:
                for line in stream:
                    digest.update(line)
                    row = json.loads(line)
                    if source == "kimi" and not row["sid"].startswith("kimi:"):
                        continue
                    assert row["sid"] not in seen, row["sid"]
                    seen.add(row["sid"])
                    assert row["fits"] and 0 < row["n_tokens"] <= 65536
                    assert len(row["ids"]) == len(row["loss"]) == row["n_tokens"]
                    assert row["ids"][0] == 128000 and row["loss"][0] == 0
                    assert set(row["loss"]) <= {0, 1} and any(row["loss"])
                    output.write(line)
                    selected_digest.update(line)
                    if source == "kimi":
                        kimi_output.write(line)
                    count += 1
                    tokens += row["n_tokens"]
                    trained += sum(row["loss"])
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            assert count == (2974 if source == "calibforge" else 4392), (source, count)
            counts[source] = {"rows": count, "tokens": tokens, "trained_tokens": trained}
            hashes[source] = {
                "path": str(path),
                "sha256": digest.hexdigest(),
                "selected_sha256": selected_digest.hexdigest(),
            }
    assert hashes["kimi"]["sha256"] == expected_kimi, "Kimi input differs from the trained vanilla model"
    assert hashes["kimi"]["selected_sha256"] == expected_kimi

    python = sys.executable
    ota = Path(os.environ["REPLAY_OTA"])
    data = root / "data"
    conversion = subprocess.check_output(
        [python, str(ota / "data/relay/sft/relay_rows_to_parquet.py"), str(rows), str(data)], text=True
    )
    (root / "parquet_census.json").write_text(conversion)
    packing = subprocess.check_output(
        [
            python,
            str(ota / "data/relay/sft/pack_epoch_steps.py"),
            str(data / "train-00000-of-00001.parquet"),
            "--seq-len",
            "65536",
            "--batch",
            "16",
        ],
        text=True,
    )
    packed = json.loads(packing.splitlines()[-1])
    (root / "pack_epoch_steps.json").write_text(json.dumps(packed) + "\n")
    report = {
        "arm": "original_relay",
        "vanilla_reference": str(vanilla),
        "matched_calibforge_tasks": len(task_counts["fdr"]),
        "sources": hashes,
        "counts": counts,
        "packing": packed,
        "recipe": {"base": "0921", "lr": 3e-4, "seed": 0, "passes": 3, "capacity": 2.0, "master_dtype": "float32"},
        "marin_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "ota_commit": subprocess.check_output(["git", "-C", str(ota), "rev-parse", "HEAD"], text=True).strip(),
    }
    (root / "prepared.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PRIORITY_DATA_OK", json.dumps(counts), json.dumps(packed), flush=True)


if __name__ == "__main__":
    main()
