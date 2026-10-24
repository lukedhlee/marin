# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare whole-row Vanilla/Relay mixtures without changing token IDs or masks."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    source: str
    index: int
    sid: str
    tokens: int
    trained: int
    sha256: str


def scan(path: Path, source: str, expected_rows: int) -> tuple[list[Row], dict]:
    before = path.stat()
    digest = hashlib.sha256()
    rows = []
    seen = set()
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            digest.update(line)
            row = json.loads(line)
            ids, loss, sid = row["ids"], row["loss"], row["sid"]
            if not isinstance(sid, str) or sid in seen:
                raise ValueError(f"Duplicate or invalid sid in {source}: {sid!r}")
            if not 0 < len(ids) <= 65536 or len(ids) != len(loss) or len(ids) != row["n_tokens"]:
                raise ValueError(f"Invalid lengths in {source}: {sid}")
            if row.get("fits") is False or ids[0] != 128000 or loss[0] != 0:
                raise ValueError(f"Invalid BOS or oversized row in {source}: {sid}")
            if any(type(x) is not int or not 0 <= x < 128256 for x in ids):
                raise ValueError(f"Invalid token in {source}: {sid}")
            if any(x not in (0, 1) for x in loss) or not any(loss):
                raise ValueError(f"Invalid loss mask in {source}: {sid}")
            seen.add(sid)
            rows.append(Row(source, index, sid, len(ids), sum(loss), hashlib.sha256(line).hexdigest()))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed while scanning: {path}")
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} {source} rows, found {len(rows)}")
    return rows, {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "rows": len(rows),
        "tokens": sum(row.tokens for row in rows),
        "trained_tokens": sum(row.trained for row in rows),
    }


def rank(row: Row, salt: str) -> bytes:
    return hashlib.sha256(f"cfmix20261023:0:{salt}:{row.source}:{row.sid}".encode()).digest()


def select(rows: list[Row], target: int) -> list[Row]:
    """Select an outcome-blind subset, approaching the target using complete rows."""
    if target <= 0 or sum(row.trained for row in rows) < target:
        raise ValueError("Insufficient supervised tokens for the balanced target")
    chosen = []
    total = 0
    for row in sorted(rows, key=lambda item: rank(item, "selection")):
        if abs(total + row.trained - target) < abs(total - target):
            chosen.append(row)
            total += row.trained
    if abs(total - target) / target > 0.005:
        raise ValueError(f"Whole-row rounding exceeds 0.5%: {total=} {target=}")
    return chosen


def write_rows(path: Path, rows: list[Row], sources: dict[str, Path]) -> dict:
    """Read only selected lines, emit shuffled complete rows with qualified IDs."""
    by_source = {source: {row.index: row for row in rows if row.source == source} for source in sources}
    # Record byte offsets rather than retaining all token arrays in memory.
    offsets = {}
    for source, source_path in sources.items():
        with source_path.open("rb") as stream:
            index = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if index in by_source[source]:
                    meta = by_source[source][index]
                    if hashlib.sha256(line).hexdigest() != meta.sha256:
                        raise ValueError(f"Selected source row changed: {source}:{meta.sid}")
                    offsets[(source, index)] = offset
                index += 1
    handles = {source: source_path.open("rb") for source, source_path in sources.items()}
    digest = hashlib.sha256()
    counts = {source: {"rows": 0, "tokens": 0, "trained_tokens": 0} for source in sources}
    try:
        with path.open("xb") as output:
            for meta in sorted(rows, key=lambda item: rank(item, "order")):
                stream = handles[meta.source]
                stream.seek(offsets[(meta.source, meta.index)])
                raw = stream.readline()
                if hashlib.sha256(raw).hexdigest() != meta.sha256:
                    raise ValueError(f"Source row changed before writing: {meta.source}:{meta.sid}")
                row = json.loads(raw)
                row["sid"] = f"{meta.source}:{meta.sid}"
                # No other field is changed, including ids, loss and outcome metadata.
                line = (json.dumps(row, separators=(",", ":")) + "\n").encode()
                output.write(line)
                digest.update(line)
                count = counts[meta.source]
                count["rows"] += 1
                count["tokens"] += meta.tokens
                count["trained_tokens"] += meta.trained
    finally:
        for stream in handles.values():
            stream.close()
    return {"sources": counts, "rows_sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expect-rows", type=int, default=2974)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    root = args.output_root
    if (root / "prepared.json").exists() or (root / "data").exists():
        raise ValueError("Refusing to replace completed or partial data preparation")
    sources = {source: args.source_root / source / f"{source}_rows.jsonl" for source in ("fdv", "fdr")}
    metadata = {}
    source_rows = {}
    for source, path in sources.items():
        source_rows[source], metadata[source] = scan(path, source, args.expect_rows)
    manifest_sha256 = None
    if args.source_manifest is not None:
        manifest_bytes = args.source_manifest.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        historical = json.loads(manifest_bytes)["per_arm"]
        for source, census in metadata.items():
            original = historical[source]
            if census["sha256"] != original["rows_sha256"] or any(
                census[key] != original[key] for key in ("rows", "tokens", "trained_tokens")
            ):
                raise ValueError(f"Source differs from the original fixed-design manifest: {source}")
    target = metadata["fdv"]["trained_tokens"] // 2
    selected = {source: select(rows, target) for source, rows in source_rows.items()}
    arms = {
        "cfmix50": [row for rows in selected.values() for row in rows],
        "cfconcat": [row for rows in source_rows.values() for row in rows],
    }
    ota = Path(os.environ["REPLAY_OTA"])
    reports = {}
    for arm, rows in arms.items():
        data = root / "data" / arm
        data.mkdir(parents=True)
        rows_path = data / "rows.jsonl"
        report = write_rows(rows_path, rows, sources)
        conversion = subprocess.check_output(
            [sys.executable, str(ota / "data/relay/sft/relay_rows_to_parquet.py"), str(rows_path), str(data)],
            text=True,
        )
        (data / "parquet_census.json").write_text(conversion)
        packing = subprocess.check_output(
            [
                sys.executable,
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
        (data / "pack_epoch_steps.json").write_text(json.dumps(packed, indent=2) + "\n")
        report["packing"] = packed
        report["selection"] = [
            {"source": row.source, "sid": row.sid, "line_sha256": row.sha256}
            for row in sorted(rows, key=lambda item: rank(item, "order"))
        ]
        reports[arm] = report
    # Check complete source hashes again before sealing; input files remain read-only.
    for source, path in sources.items():
        _, after = scan(path, source, args.expect_rows)
        if after != metadata[source]:
            raise ValueError(f"Source changed during preparation: {source}")
    prepared = {
        "sources": metadata,
        "source_manifest_sha256": manifest_sha256,
        "balanced_reference": "Vanilla supervised tokens per corpus pass",
        "balanced_target_per_source": target,
        "selection_seed": 0,
        "selection_reads_outcomes": False,
        "whole_row_rounding_limit": 0.005,
        "arms": reports,
        "marin_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "ota_commit": subprocess.check_output(["git", "-C", str(ota), "rev-parse", "HEAD"], text=True).strip(),
    }
    (root / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    print("CF_MIXED_DATA_OK", json.dumps({arm: report["sources"] for arm, report in reports.items()}), flush=True)


if __name__ == "__main__":
    main()
