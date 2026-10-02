#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# One relay SFT run end to end, from a login node (inside tmux: it only submits jobs and waits on them):
#   prep (1 node) and import (4 nodes) side by side -> train (4 nodes) -> one HF export per kept checkpoint (4 nodes
#   each, side by side). Each step is skipped when its artifact exists, so rerunning resumes.
#
#   RELAY_CLUSTER=jupiter RELAY_ACCOUNT=<project> RELAY_ROOT=<data root> \
#   BASE=<HF dir of the base> PARQUET=<train-00000-of-00001.parquet> NAME=<run name> bash run_arm.sh
#
# Optional: ROWS=<rendered rows jsonl> to build PARQUET first; INIT=<native init> (default: the base's import under
# RELAY_ROOT/inits, made once and shared by every run from that base); EPOCHS (3); STEPS (train only the first STEPS
# of the schedule); WALL (train time limit, default 08:00:00). Costs at the recipe's layout: import ~0.5 node-h,
# training ~7 s per step on 4 nodes (a 1,200-step run is ~9.5 node-h), each export ~0.5 node-h.
set -uo pipefail
: "${BASE:?HF dir of the base}" "${PARQUET:?Parquet shard path}" "${NAME:?run name}"
source "$(cd "$(dirname "$0")" && pwd)/cluster.sh"
RUN=$RELAY_ROOT/runs/$NAME
CACHE=$RUN/cache
OUTPUT=$RUN/train
INIT=${INIT:-$RELAY_ROOT/inits/$(basename "$BASE")/step-0}
LOGD=$RELAY_ROOT/logs/$NAME
mkdir -p "$LOGD" "$RUN" "$(dirname "$INIT")"
SUBMIT=(sbatch --parsable -A "$RELAY_ACCOUNT" -p "$RELAY_PARTITION" --gres=gpu:4 --export=ALL)
export MARIN_ROOT RELAY_CLUSTER RELAY_ROOT RELAY_PYTHON BASE PARQUET CACHE INIT OUTPUT RUN_ID=$NAME

say() { echo "[$(date -u +%FT%TZ)] [$NAME] $*" | tee -a "$LOGD/run_arm.log"; }
die() { say "RUN_FAILED: $*"; exit 1; }
submit() { "${SUBMIT[@]}" "$@" | tail -1; }   # some clusters print a banner before the job id
wait_job() {  # $1 job id; succeeds when the job COMPLETED
  local state
  while squeue -h -j "$1" -t PD,R,CF,S,RQ,RS 2>/dev/null | grep -q .; do sleep 60; done
  for _ in 1 2 3 4 5; do
    state=$(sacct -j "$1" -X -n -o State%20 2>/dev/null | head -1 | awk '{print $1}')
    [ -n "$state" ] && break
    sleep 20
  done
  say "job $1 ended ${state:-unknown}"
  [ "$state" = COMPLETED ]
}

say "RUN_START marin=$(git -C "$MARIN_ROOT" rev-parse --short HEAD) base=$BASE parquet=$PARQUET init=$INIT"
[ -f "$BASE/config.json" ] && [ -f "$BASE/tokenizer.json" ] && [ -f "$BASE/chat_template.jinja" ] || die "no HF export at $BASE"

prep=; import=
if [ ! -f "$CACHE/relay_cache.json" ]; then
  [ -f "$PARQUET" ] || [ -n "${ROWS:-}" ] || die "no Parquet at $PARQUET and no ROWS to build it from"
  prep=$(submit -c "$RELAY_NODE_CPUS" -o "$LOGD/prep.%j.log" "$RELAY_SLURM_DIR/prep.sbatch") || die "prep submit"
  say "prep job $prep"
fi
if [ ! -f "$INIT/snowball_base.json" ]; then
  [ -e "$INIT" ] && die "$INIT exists without snowball_base.json (an interrupted import); remove it"
  import=$(submit -o "$LOGD/import.%j.log" "$RELAY_SLURM_DIR/import.sbatch") || die "import submit"
  say "import job $import"
fi
[ -z "$prep" ] || wait_job "$prep" || die "prep job $prep ($LOGD/prep.$prep.log)"
[ -z "$import" ] || { wait_job "$import" && [ -f "$INIT/snowball_base.json" ]; } || die "import job $import"
say "cache: $(tr -d '\n ' < "$CACHE/relay_cache.json" | head -c 400)"

if [ ! -f "$LOGD/train.done" ]; then
  [ -d "$OUTPUT/checkpoints" ] && export RESUME=1
  train=$(submit -t "${WALL:-08:00:00}" -o "$LOGD/train.%j.log" "$RELAY_SLURM_DIR/train.sbatch") || die "train submit"
  say "train job $train (resume=${RESUME:-0})"
  wait_job "$train" || die "train job $train ($LOGD/train.$train.log)"
  touch "$LOGD/train.done"
fi

declare -A exports=()
for checkpoint in "$OUTPUT"/checkpoints/step-*; do
  [ -f "$checkpoint/metadata.json" ] || continue
  export_dir=$RUN/export-$(basename "$checkpoint")-hf-bf16
  [ -f "$export_dir/config.json" ] && continue
  job=$(CHECKPOINT=$checkpoint EXPORT=$export_dir submit -o "$LOGD/export.%j.log" "$RELAY_SLURM_DIR/export.sbatch") \
    || die "export submit for $checkpoint"
  exports[$job]=$export_dir
  say "export job $job -> $export_dir"
done
for job in "${!exports[@]}"; do
  wait_job "$job" && grep -q EXPORT_CHECK_OK "$LOGD/export.$job.log" || die "export job $job"
done
say "RUN_DONE exports: $(ls -d "$RUN"/export-*-hf-bf16 2>/dev/null | tr '\n' ' ')"
