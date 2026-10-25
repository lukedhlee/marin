#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Guard this replay's submissions against the shared Horizon SU ceiling."""

import fcntl
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def hours(value: str) -> float:
    days, _, clock = value.rpartition("-")
    fields = [int(x) for x in (clock or value).split(":")]
    if len(fields) == 3:
        result = fields[0] + fields[1] / 60 + fields[2] / 3600
    elif len(fields) == 2:
        result = fields[0] / 60 + fields[1] / 3600
    else:
        result = fields[0] / 60
    return int(days or 0) * 24 + result


def resources(args: list[str]) -> tuple[int, float]:
    defaults = []
    for arg in reversed(args):
        path = Path(arg)
        if not arg.startswith("-") and path.is_file():
            defaults = [
                token
                for line in path.read_text().splitlines()
                if line.startswith("#SBATCH ")
                for token in shlex.split(line[8:])
            ]
            break
    nodes = wall = None
    tokens = iter(defaults + args)
    for token in tokens:
        if token in ("-N", "--nodes"):
            nodes = next(tokens)
        elif token.startswith("--nodes="):
            nodes = token.split("=", 1)[1]
        elif token.startswith("-N"):
            nodes = token[2:]
        elif token in ("-t", "--time"):
            wall = next(tokens)
        elif token.startswith("--time="):
            wall = token.split("=", 1)[1]
        elif token.startswith("-t"):
            wall = token[2:]
    if nodes is None or wall is None:
        raise ValueError("Replay submissions must specify nodes and wall time.")
    return int(nodes), hours(wall)


def main() -> None:
    args = sys.argv[1:]
    nodes, wall = resources(args)
    root = Path(os.environ["FP32_REPLAY_ROOT"])
    with (root / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = subprocess.check_output(["/usr/bin/squeue", "-A", "CCR24067", "-h", "-o", "%u|%D|%l"], text=True)
        booked = {}
        for line in queue.splitlines():
            user, count, duration = line.split("|")
            booked[user] = booked.get(user, 0) + int(count) * hours(duration)
        requested = nodes * wall
        mine = booked.get(os.environ["USER"], 0)
        total = sum(booked.values())
        reserved_nodes = sum(
            int(line)
            for line in subprocess.check_output(
                ["/usr/bin/squeue", "-u", os.environ["USER"], "-h", "-o", "%D"], text=True
            ).splitlines()
            if line.strip()
        )
        if mine + requested > 700 or total + requested > 1000 or reserved_nodes + nodes > 100:
            print(
                f"Waiting for SU room: ours={mine:.2f}, project={total:.2f}, request={requested:.2f}; "
                f"300 reserved for colleagues; reserved_nodes={reserved_nodes}, requested_nodes={nodes}",
                file=sys.stderr,
            )
            raise SystemExit(75)
        record = {
            "utc": datetime.now(UTC).isoformat(),
            "booked": booked,
            "requested_su": requested,
            "args": args,
            "colleague_reserve": 300,
        }
        # Keep site output intact: Horizon prints a banner even with --parsable.
        result = subprocess.run(["/usr/bin/sbatch", *args], capture_output=True, text=True)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        with (root / "submissions.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
