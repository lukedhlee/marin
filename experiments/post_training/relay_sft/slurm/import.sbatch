#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# HF export of the base (BASE) -> native step-0 init (INIT) with pending_qb_betas = -router_bias. Four CPU ranks,
# one per node, each holding a quarter of the 134 GB source and of the converted layout; the GPUs idle.
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00
#SBATCH --job-name=relay-sft-import
set -euo pipefail
: "${MARIN_ROOT:?}" "${BASE:?}" "${INIT:?}"
source "$MARIN_ROOT/experiments/post_training/relay_sft/slurm/cluster.sh"
relay_compute_env
export JAX_PLATFORMS=cpu
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task="$RELAY_NODE_CPUS" --cpu-bind=cores \
  "$RELAY_PYTHON" -m experiments.post_training.relay_sft.relay_sft import-hf --hf-checkpoint "$BASE" --output-path "$INIT"
