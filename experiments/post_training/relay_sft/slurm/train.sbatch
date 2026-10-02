#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# The training run: four one-GPU ranks per node (Levanter's Slurm cluster gives each local rank its own GPU), every
# rank running relay_sft.py train. A watchdog cancels the job after STALL_SECONDS without a new progress line, so a
# wedged collective does not hold the nodes until the time limit. run_arm.sh submits it with --export=ALL.
#
# Inputs: INIT, CACHE, BASE (tokenizer), OUTPUT, RUN_ID; EPOCHS (3), STEPS (unset = every epoch), RESUME=1 to
# continue OUTPUT from its latest checkpoint.
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --job-name=relay-sft-train
set -uo pipefail
: "${MARIN_ROOT:?}" "${INIT:?}" "${CACHE:?}" "${BASE:?}" "${OUTPUT:?}" "${RUN_ID:?}"
source "$MARIN_ROOT/experiments/post_training/relay_sft/slurm/cluster.sh"
relay_compute_env
relay_gpu_env
STALL_SECONDS=${STALL_SECONDS:-1800}
LOG=$RELAY_ROOT/logs/train-$RUN_ID-$SLURM_JOB_ID.log
mkdir -p "$(dirname "$LOG")"
ARGS=(--init-checkpoint-path "$INIT" --cache-path "$CACHE" --tokenizer "$BASE" --output-path "$OUTPUT"
      --run-id "$RUN_ID" --epochs "${EPOCHS:-3}" ${STEPS:+--steps "$STEPS"})
[ "${RESUME:-0}" = 1 ] && ARGS+=(--resume)
CLI=("$RELAY_PYTHON" -m experiments.post_training.relay_sft.relay_sft train)

JAX_PLATFORMS=cpu "${CLI[@]}" "${ARGS[@]}" --check-only || exit 4

watchdog() {  # progress lines appear only when a step completes; their count standing still means a stall
  local seen=-1 count since=$(date +%s)
  while sleep 60; do
    count=$(grep -c "Progress on:train" "$LOG" 2>/dev/null)
    if [ "${count:-0}" != "$seen" ]; then seen=$count; since=$(date +%s); continue; fi
    if [ $(( $(date +%s) - since )) -ge "$STALL_SECONDS" ]; then
      echo "WATCHDOG: no training progress for ${STALL_SECONDS}s after $seen progress lines; cancelling"
      scancel "$SLURM_JOB_ID"
      return
    fi
  done
}
watchdog & WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null' EXIT

srun --ntasks="$SLURM_NTASKS" --ntasks-per-node=4 --cpus-per-task=$((RELAY_NODE_CPUS / 4)) --gpu-bind=none \
  --cpu-bind=cores "${CLI[@]}" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "RELAY_TRAIN_EXIT rc=$RC"
exit "$RC"
