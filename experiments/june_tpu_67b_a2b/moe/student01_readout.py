# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare to the completed baseline with aliases that cannot collapse repetition IDs."""

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root, ota = map(Path, sys.argv[1:])
    scratch = Path(f"/scratch/11584/{os.environ['USER']}")
    baseline = scratch / "experiments/relay-kimi-fp32-capacity2-20261021/readout_links"
    jobs = scratch / "experiments/sft_eval/tb2_jobs"
    aliases = root / "readout_links"
    aliases.mkdir(exist_ok=True)
    for shard in ("tb21", "swe_s0", "swe_s1", "tblite_s0", "tblite_s1"):
        family = "swe" if shard.startswith("swe") else "terminal"
        for rep in (1, 2):
            for tag in ("rkimicf2", "rstudent01"):
                if tag == "rkimicf2":
                    source = baseline / f"{shard}_{tag}_r{rep}_kc2_20261021"
                else:
                    source = jobs / f"{shard}_{tag}_r{rep}_student01{family}_20261025"
                assert source.is_dir(), f"Missing prescribed evaluation: {source}"
                link = aliases / f"{shard}_{tag}_r{rep}_st01_20261025"
                if link.is_symlink():
                    assert link.resolve() == source.resolve()
                else:
                    link.symlink_to(source.resolve(), target_is_directory=True)
    subprocess.run(
        [
            sys.executable,
            str(ota / "data/r2egym/horizon/sft/eval_readout.py"),
            "--tags",
            "rkimicf2",
            "rstudent01",
            "--jobs",
            str(aliases),
            "--day",
            "st01_20261025",
            "--out",
            str(root / "readout.json"),
        ],
        check=True,
    )
    readout = json.loads((root / "readout.json").read_text())
    coverage = {}
    for tag in ("rkimicf2", "rstudent01"):
        for family, expected in (("tb21", 88), ("swe", 100), ("tblite", 100)):
            reps = readout[f"{tag}/{family}"]["reps"]
            assert set(reps) == {"1", "2"}, f"Repetition parsing failed: {tag}/{family}"
            for rep, value in reps.items():
                finished = value["counts"]["trials"]
                coverage[f"{tag}/{family}/{rep}"] = {
                    "expected": expected,
                    "finished": finished,
                    "scored": value["n"],
                    "adequate": finished == expected and value["n"] / expected >= 0.9,
                }
    (root / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print("Coverage adequate for all cells:", all(v["adequate"] for v in coverage.values()))


if __name__ == "__main__":
    main()
