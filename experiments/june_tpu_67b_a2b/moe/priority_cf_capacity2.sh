#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# CalibForge-only vanilla (fdv) and original Relay (fdr) with both fixes: FP32 masters and expert capacity 2.0.
# Reuses the fixed-design arms' exact packed rows and validated token caches (relay-fixed-design-20261021), so the only
# change from the capacity-1.0 fdv/fdr pair is this checkout's capacity. Recipe as train_fd_arms.sh: init 09-21, LR 3e-4,
# 3 passes, warmup round(5 %), batch 16 x 65,536 on 4 nodes, seed 0, one kept checkpoint per pass, FP32 check of each.
# Then the final-pass exports, and both models x 3 reps of TB2.1 / SWE100 / TB-lite (one 5-node group each).
# Tags cfv2 / cfr2 (no leading "r": eval_readout.py takes the rep from the last "_r" in a run name).
set -euo pipefail
unset LD_PRELOAD
: "${MARIN_ROOT:?}"
: "${FP32_REPLAY_ROOT:?}"
: "${REPLAY_OTA:?}"
R=$FP32_REPLAY_ROOT
F=/scratch/11584/$USER/experiments/relay-fixed-design-20261021/sft
MOE=$MARIN_ROOT/experiments/june_tpu_67b_a2b/moe
BASE=/scratch/11584/$USER/models/grug-datakit-sft-20260921
REV=fd20261021
declare -A TAG=([fdv]=cfv2 [fdr]=cfr2) WALL=([fdv]=02:00:00 [fdr]=02:15:00)
export MARIN_PYTHON=$HOME/snowball/envs/marin-grug-sft/bin/python
export PYTHONPATH="$MARIN_ROOT:$MARIN_ROOT/lib/levanter/src:$MARIN_ROOT/lib/haliax/src:$MARIN_ROOT/lib/marin/src:$MARIN_ROOT/lib/fray/src:$MARIN_ROOT/lib/zephyr/src:$MARIN_ROOT/lib/iris/src:$MARIN_ROOT/lib/rigging/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$MARIN_ROOT"
mkdir -p "$R/bin" "$R/logs" "$R/training"
exec 9>"$R/driver.lock"
flock -n 9 || { echo "Another driver owns $R"; exit 1; }
trap 'echo "FAILED at line $LINENO; inspect driver.log and phase logs" > "$R/STATUS"' ERR
[ -z "$(git status --porcelain --untracked-files=no)" ]
grep -q '^_DEFAULT_EP_CAPACITY_FACTOR = 2.0$' "$MOE/model.py"
grep -q '"fdv", "fdr", "fdj2", "fdk1", "fddf"' "$MOE/vista_snowball_chat.py"
ln -sfn "$MOE/replay_sbatch.py" "$R/bin/sbatch"
export PATH="$R/bin:$PATH"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$R/driver.log" > "$R/STATUS"; }
submit() {
  local output rc
  while :; do
    set +e; output=$(sbatch "$@" 2>&1); rc=$?; set -e
    if [ "$rc" = 75 ]; then echo "$output" >&2; sleep 60; continue; fi
    [ "$rc" = 0 ] || { echo "$output" >&2; return "$rc"; }
    echo "$output" | sed -nE 's/Submitted batch job ([0-9]+).*/\1/p' | tail -1
    return
  done
}
wait_ok() {  # $1 job, $2 log, $3 marker
  while squeue -h -j "$1" -t PD,R,CF,S,RQ,RS | grep -q .; do sleep 30; done
  grep -q "$3" "$2"
}
check_dtype() {
  "$MARIN_PYTHON" - "$1" <<'PY'
import json,sys,tensorstore as ts
from pathlib import Path
p=Path(sys.argv[1]);manifest=json.loads((p/'manifest.json').read_text())
params=[a for a in manifest['arrays'] if a['path'].startswith('params/')]
assert params and all(a['dtype']=='float32' for a in params),[(a['path'],a['dtype']) for a in params]
a=ts.open({'driver':'zarr3','kvstore':{'driver':'ocdbt','base':{'driver':'file','path':str(p)+'/'},'path':'params/token_embed'}},open=True,read=True,write=False).result()
assert a.dtype.name=='float32',a.dtype
print('FP32_CHECKPOINT_OK',p,flush=True)
PY
}
select_arm() {
  local arm=$1 D=$F/data/$1
  epoch=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["pack_epoch_steps"])' "$D/pack_epoch_steps.json")
  steps=$((3 * epoch)); warmup=$(((steps * 5 + 50) / 100)); [ "$warmup" -ge 2 ] || warmup=2
  export SNOWBALL_SCRATCH=$R SNOWBALL_INIT=/scratch/11584/$USER/snowball-sft/experiments/snowball-base-inits/init-dk0921-step0
  export SNOWBALL_TOKENIZER=$BASE SNOWBALL_SEED=0 SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16
  export SNOWBALL_NODES=4 SNOWBALL_DEVICES=16 SNOWBALL_LR=3e-4 SNOWBALL_STAGE=relay_$arm
  export SNOWBALL_STEPS=$steps SNOWBALL_SCHEDULE_STEPS=$steps SNOWBALL_EPOCH_STEPS=$epoch SNOWBALL_WARMUP=$warmup
  export SNOWBALL_CACHE=$F/cache-relay_$arm-v1 SNOWBALL_PARQUET_LIST=$D/parquet.list
  export SNOWBALL_DATASET_ID=private/relay-calibforge-sft-$arm SNOWBALL_DATASET_REVISION=$REV
  export SNOWBALL_OUTPUT=$R/training/$arm SNOWBALL_RUN_ID=fp32-cap2-$arm-0921-20261022
  export SNOWBALL_SAVE_INTERVAL_MIN=100000 SNOWBALL_KEEP_PER_EPOCH=1 SNOWBALL_PROBE_MEMORY=0
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
  unset SNOWBALL_PROBE_STEPS SNOWBALL_KEEP_EVERY SNOWBALL_RESUME_STEP SNOWBALL_TAIL_FRACTION
  unset SNOWBALL_TAIL_REF SNOWBALL_TAIL_RAMP SNOWBALL_TAIL_SCORE_OUT SNOWBALL_XLA_EXTRA
}
say "START CalibForge-only fdv/fdr FP32 capacity2 commit=$(git rev-parse HEAD)"
# The capacity-1.0 arms' caches, validated by their prep jobs; read only here.
for arm in fdv fdr; do
  [ -f "$F/cache-relay_$arm-v1/train/.stats.json" ] && grep -q PREP_DONE "$F"/logs/prep-$arm.*.log
done
for arm in fdv fdr; do
  select_arm "$arm"
  if [ ! -f "$R/$arm.train.job" ]; then
    [ ! -e "$SNOWBALL_OUTPUT" ]
    j=$(submit -A CCR24067 -p debug -N 4 -t "${WALL[$arm]}" -J "cf2_$arm" -o "$R/logs/train-$arm.%j.log" --export=ALL \
        "$MOE/horizon_snowball_guarded.sbatch")
    [ -n "$j" ]; echo "$j" > "$R/$arm.train.job"
    say "train $arm submitted job$j steps=$steps epoch=$epoch warmup=$warmup wall=${WALL[$arm]}"
  fi
done
for arm in fdv fdr; do
  select_arm "$arm"
  if [ ! -f "$R/$arm.train.ok" ]; then
    j=$(cat "$R/$arm.train.job")
    wait_ok "$j" "$R/logs/train-$arm.$j.log" 'GUARDED_RUN_EXIT rc=0'
    for s in "$epoch" "$((2 * epoch))" "$steps"; do check_dtype "$SNOWBALL_OUTPUT/checkpoints/step-$s"; done
    touch "$R/$arm.train.ok"; say "train $arm OK, FP32 checkpoints step $epoch/$((2 * epoch))/$steps"
  fi
done
declare -A EXP=()
for arm in fdv fdr; do
  select_arm "$arm"
  export SNOWBALL_EXPORT_CHECKPOINT=$SNOWBALL_OUTPUT/checkpoints/step-$steps
  export SNOWBALL_EXPORT_OUTPUT=$SNOWBALL_OUTPUT/export-step$steps-hf-bf16
  export SNOWBALL_EXPORT_TOKENIZER=$BASE SNOWBALL_EXPORT_BASE=$BASE
  EXP[$arm]=$SNOWBALL_EXPORT_OUTPUT
  if [ ! -f "$R/$arm.export.ok" ] && [ ! -f "$R/$arm.export.job" ]; then
    j=$(submit -A CCR24067 -p debug -N 4 -t 00:25:00 -J "cf2_export_$arm" -o "$R/logs/export-$arm.%j.log" --export=ALL \
        "$MOE/horizon_snowball_export.sbatch")
    [ -n "$j" ]; echo "$j" > "$R/$arm.export.job"; say "export $arm submitted job$j"
  fi
done
for arm in fdv fdr; do
  if [ ! -f "$R/$arm.export.ok" ]; then
    j=$(cat "$R/$arm.export.job")
    wait_ok "$j" "$R/logs/export-$arm.$j.log" EXPORT_CHECK_OK
    touch "$R/$arm.export.ok"; say "export $arm OK"
  fi
done
export MODELS="${TAG[fdv]}=${EXP[fdv]} ${TAG[fdr]}=${EXP[fdr]}" REPS="1 2 3" GROUPED=1 MAXJOBS=20 SERVE_TIME=03:00:00
export HARBOR_SRC=$HOME/snowball/harbor-p0924/src HARBOR_SHA=761fb516 POLICY_FILE=tb2_marin_policy_0924_65k16k.yaml
export DAY=cf2_20261022 SETS="tb21 swe_s0 swe_s1 tblite_s0 tblite_s1"
if [ ! -f "$R/eval.ok" ]; then
  # 2 models x 3 reps x 5 sets x 16 concurrent trials = 480 sandboxes (Luke 2026-10-22: third rep). Admission threshold DAYTONA_CAP (default 850;
  # Luke 2026-10-22: the org can take 1,000 more, so this run uses 1,850).
  while :; do
    ds=$(python3 "$REPLAY_OTA/data/r2egym/horizon/sft/daytona_started.py" 2>/dev/null || true)
    [[ "$ds" =~ ^[0-9]+$ ]] && [ $((ds + 480)) -le "${DAYTONA_CAP:-850}" ] && break
    say "Waiting for Daytona room before eval (started=$ds)"; sleep 60
  done
  say "EVAL_START"
  bash "$REPLAY_OTA/data/r2egym/horizon/sft/eval_sft.sh" > "$R/logs/eval.log" 2>&1
  touch "$R/eval.ok"
fi
python3 "$REPLAY_OTA/data/r2egym/horizon/sft/eval_readout.py" --tags cfv2 cfr2 fdv fdr --pairs-with cfv2 cfr2 \
  --out "$R/readout.json" > "$R/readout.txt"
say "READOUT_WRITTEN: audit coverage and errors before comparing scores"
