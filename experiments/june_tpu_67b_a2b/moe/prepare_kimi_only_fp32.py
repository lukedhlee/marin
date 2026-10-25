# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Freeze the same filtered Kimi rows used by both corrected-stack mixtures."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(os.environ["FP32_REPLAY_ROOT"])
    scratch = Path(f"/scratch/11584/{os.environ['USER']}")
    sources = {
        "kimi": scratch / "experiments/vanilla-kimi-fp32-capacity2-20261021/kimi_rows.jsonl",
    }
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "prepared.json").exists(), "Preparation is already complete"
    rows = root / "rows.jsonl"
    assert not rows.exists(), "Refusing to replace partial or completed rows"
    counts = {}
    hashes = {}
    seen = set()
    with rows.open("xb") as output:
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
                    count += 1
                    tokens += row["n_tokens"]
                    trained += sum(row["loss"])
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            assert count == 4392, (source, count)
            assert selected_digest.hexdigest() == "f76420a79f01f7ddba8a343f9f13dea7330bc9bb9e7b95c35ffefd3ade1276b1"
            assert (tokens, trained) == (65972582, 21571114)
            counts[source] = {"rows": count, "tokens": tokens, "trained_tokens": trained}
            hashes[source] = {
                "path": str(path),
                "sha256": digest.hexdigest(),
                "selected_sha256": selected_digest.hexdigest(),
            }

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
        "sources": hashes,
        "counts": counts,
        "packing": packed,
        "recipe": {
            "base": "0921",
            "lr": 3e-4,
            "seed": 0,
            "passes": 3,
            "capacity": 2.0,
            "master_dtype": "float32",
            "stage": "relay_kimifp2",
        },
        "marin_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "ota_commit": subprocess.check_output(["git", "-C", str(ota), "rev-parse", "HEAD"], text=True).strip(),
    }
    (root / "prepared.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PRIORITY_DATA_OK", json.dumps(counts), json.dumps(packed), flush=True)


if __name__ == "__main__":
    main()
