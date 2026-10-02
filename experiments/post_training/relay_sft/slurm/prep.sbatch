#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# One node: rendered rows (ROWS, jsonl) -> Parquet when ROWS is set, then PARQUET -> the Levanter cache at CACHE
# with its relay_cache.json. run_arm.sh submits it with --export=ALL; the base's HF dir BASE is the tokenizer.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=relay-sft-prep
set -euo pipefail
: "${MARIN_ROOT:?}" "${BASE:?}" "${PARQUET:?}" "${CACHE:?}"
source "$MARIN_ROOT/experiments/post_training/relay_sft/slurm/cluster.sh"
relay_compute_env
export JAX_PLATFORMS=cpu
CLI=("$RELAY_PYTHON" -m experiments.post_training.relay_sft.relay_sft)
if [ -n "${ROWS:-}" ] && [ ! -f "$PARQUET" ]; then
  "${CLI[@]}" rows-to-parquet "$ROWS" "$(dirname "$PARQUET")"
fi
[ -f "$CACHE/relay_cache.json" ] || "${CLI[@]}" prepare-cache --parquet "$PARQUET" --tokenizer "$BASE" --cache-path "$CACHE"
echo PREP_DONE
