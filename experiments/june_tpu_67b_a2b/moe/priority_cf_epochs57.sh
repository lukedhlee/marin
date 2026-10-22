#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Matched capacity-2 CalibForge runs: one seven-epoch schedule, kept at epochs five and seven.
set -euo pipefail
unset LD_PRELOAD
: "${MARIN_ROOT:?}"
: "${FP32_REPLAY_ROOT:?}"
: "${REPLAY_OTA:?}"
R=$FP32_REPLAY_ROOT
F=/scratch/11584/$USER/experiments/relay-fixed-design-20261021/sft
MOE=$MARIN_ROOT/experiments/june_tpu_67b_a2b/moe
BASE=/scratch/11584/$USER/models/grug-datakit-sft-20260921
declare -A WALL=([fdv]=04:00:00 [fdr]=04:30:00)
declare -A TAG=([fdv5]=cv5 [fdr5]=cr5 [fdv7]=cv7 [fdr7]=cr7)
export MARIN_PYTHON=$HOME/snowball/envs/marin-grug-sft/bin/python
export PYTHONPATH="$MARIN_ROOT:$MARIN_ROOT/lib/levanter/src:$MARIN_ROOT/lib/haliax/src:$MARIN_ROOT/lib/marin/src:$MARIN_ROOT/lib/fray/src:$MARIN_ROOT/lib/zephyr/src:$MARIN_ROOT/lib/iris/src:$MARIN_ROOT/lib/rigging/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CC=gcc CXX=g++
cd "$MARIN_ROOT"
mkdir -p "$R/bin" "$R/logs" "$R/training"
exec 9>"$R/driver.lock"
flock -n 9 || { echo "Another driver owns $R"; exit 1; }
trap 'echo "FAILED at line $LINENO; inspect phase logs" > "$R/STATUS"' ERR
[ -z "$(git status --porcelain --untracked-files=no)" ]
grep -q '^_DEFAULT_EP_CAPACITY_FACTOR = 2.0$' "$MOE/model.py"
ln -sfn "$MOE/replay_sbatch.py" "$R/bin/sbatch"
export PATH="$R/bin:$PATH"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$R/driver.log" > "$R/STATUS"; }
submit() {
  local output rc nodes
  while :; do
    nodes=$(squeue -u "$USER" -h -o %D | awk '{n+=$1} END {print n+0}')
    if [ "$nodes" -gt 96 ]; then sleep 60; continue; fi
    set +e; output=$(sbatch "$@" 2>&1); rc=$?; set -e
    if [ "$rc" = 75 ]; then echo "$output" >&2; sleep 60; continue; fi
    [ "$rc" = 0 ] || { echo "$output" >&2; return "$rc"; }
    echo "$output" | sed -nE 's/Submitted batch job ([0-9]+).*/\1/p' | tail -1
    return
  done
}
wait_ok() {
  while squeue -h -j "$1" -t PD,R,CF,S,RQ,RS | grep -q .; do sleep 30; done
  grep -q "$3" "$2"
}
check_dtype() {
  "$MARIN_PYTHON" - "$1" <<'PY'
import json, sys, tensorstore as ts
from pathlib import Path
p = Path(sys.argv[1])
arrays = json.loads((p / 'manifest.json').read_text())['arrays']
params = [a for a in arrays if a['path'].startswith('params/')]
assert params and all(a['dtype'] == 'float32' for a in params)
a = ts.open({'driver': 'zarr3', 'kvstore': {'driver': 'ocdbt', 'base': {'driver': 'file', 'path': str(p) + '/'},
                                                        'path': 'params/token_embed'}}, open=True, read=True, write=False).result()
assert a.dtype.name == 'float32', a.dtype
print('FP32_CHECKPOINT_OK', p, flush=True)
PY
}
select_arm() {
  local arm=$1 D=$F/data/$1
  epoch=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["pack_epoch_steps"])' "$D/pack_epoch_steps.json")
  steps=$((7 * epoch)); warmup=$(((steps * 5 + 50) / 100))
  export SNOWBALL_SCRATCH=$R SNOWBALL_INIT=/scratch/11584/$USER/snowball-sft/experiments/snowball-base-inits/init-dk0921-step0
  export SNOWBALL_TOKENIZER=$BASE SNOWBALL_SEED=0 SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16
  export SNOWBALL_NODES=4 SNOWBALL_DEVICES=16 SNOWBALL_LR=3e-4 SNOWBALL_STAGE=relay_$arm
  export SNOWBALL_STEPS=$steps SNOWBALL_SCHEDULE_STEPS=$steps SNOWBALL_EPOCH_STEPS=$epoch SNOWBALL_WARMUP=$warmup
  export SNOWBALL_CACHE=$F/cache-relay_$arm-v1 SNOWBALL_PARQUET_LIST=$D/parquet.list
  export SNOWBALL_DATASET_ID=private/relay-calibforge-sft-$arm SNOWBALL_DATASET_REVISION=fd20261021
  export SNOWBALL_OUTPUT=$R/training/$arm SNOWBALL_RUN_ID=fp32-cap2-e7-$arm-0921-20261022
  # Keep only epoch five and the forced final epoch-seven checkpoint.
  export SNOWBALL_KEEP_EVERY=$((5 * epoch)) SNOWBALL_SAVE_INTERVAL_MIN=100000 SNOWBALL_PROBE_MEMORY=0
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
  unset SNOWBALL_PROBE_STEPS SNOWBALL_KEEP_PER_EPOCH SNOWBALL_RESUME_STEP SNOWBALL_TAIL_FRACTION
  unset SNOWBALL_TAIL_REF SNOWBALL_TAIL_RAMP SNOWBALL_TAIL_SCORE_OUT SNOWBALL_XLA_EXTRA
}
for arm in fdv fdr; do
  [ -f "$F/cache-relay_$arm-v1/train/.stats.json" ] && grep -q PREP_DONE "$F"/logs/prep-$arm.*.log
done
python3 - "$R" "$F" "$MARIN_ROOT" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
r, f, code = map(Path, sys.argv[1:])
arms = {}
for arm, expected in [('fdv', 137), ('fdr', 168)]:
    d = f / 'data' / arm
    epoch = json.loads((d / 'pack_epoch_steps.json').read_text())['pack_epoch_steps']
    assert epoch == expected
    arms[arm] = dict(epoch_steps=epoch, total_steps=7 * epoch, export_steps=[5 * epoch, 7 * epoch],
                     warmup_steps=(7 * epoch * 5 + 50) // 100, lr=3e-4, seed=0,
                     parquet_list_sha256=hashlib.sha256((d / 'parquet.list').read_bytes()).hexdigest())
manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), arms=arms,
                code_commit=subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip(),
                design='Fresh seven-epoch schedules from the same 09-21 init, retaining intermediate epoch five and final epoch seven; not a dedicated five-epoch cooldown.',
                capacity_factor=2.0, param_dtype='float32', training_nodes_per_arm=4,
                eval_reps=[1, 2, 3], eval_benchmarks=['tb21', 'swe', 'tblite'],
                max_eval_concurrency=960, sandbox_admission_limit=2000,
                booked_node_hours=dict(training=34, exports=20 / 3, eval=180),
                authorization='Luke: extend to 5 and 7 epochs for both; up to 2000 concurrent sandboxes, 2026-10-22')
p = r / 'manifest.json'
if p.exists():
    old = json.loads(p.read_text())
    assert old['arms'] == arms and old['code_commit'] == manifest['code_commit']
else:
    p.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
if [ "${DRY_RUN:-0}" = 1 ]; then say "DRY_RUN_OK: no jobs submitted"; exit 0; fi
say "START epochs 5/7, FP32 capacity2 commit=$(git rev-parse HEAD)"
for arm in fdv fdr; do
  select_arm "$arm"
  if [ ! -f "$R/$arm.train.job" ]; then
    [ ! -e "$SNOWBALL_OUTPUT" ]
    j=$(submit -A CCR24067 -p debug -N 4 -t "${WALL[$arm]}" -J "cf2e7_$arm" -o "$R/logs/train-$arm.%j.log" --export=ALL \
        "$MOE/horizon_snowball_guarded.sbatch")
    [ -n "$j" ]; echo "$j" > "$R/$arm.train.job"
    say "train $arm job$j steps=$steps epoch=$epoch warmup=$warmup wall=${WALL[$arm]}"
  fi
done
for arm in fdv fdr; do
  select_arm "$arm"
  if [ ! -f "$R/$arm.train.ok" ]; then
    j=$(cat "$R/$arm.train.job")
    wait_ok "$j" "$R/logs/train-$arm.$j.log" 'GUARDED_RUN_EXIT rc=0'
    for e in 5 7; do check_dtype "$SNOWBALL_OUTPUT/checkpoints/step-$((e * epoch))"; done
    touch "$R/$arm.train.ok"; say "train $arm OK, FP32 checkpoints epochs 5/7"
  fi
done
MODELS=
for arm in fdv fdr; do
  select_arm "$arm"
  for e in 5 7; do
    s=$((e * epoch)); cell=$arm$e
    export SNOWBALL_EXPORT_CHECKPOINT=$SNOWBALL_OUTPUT/checkpoints/step-$s
    export SNOWBALL_EXPORT_OUTPUT=$SNOWBALL_OUTPUT/export-step$s-hf-bf16
    export SNOWBALL_EXPORT_TOKENIZER=$BASE SNOWBALL_EXPORT_BASE=$BASE
    MODELS+="${TAG[$cell]}=$SNOWBALL_EXPORT_OUTPUT "
    if [ ! -f "$R/$cell.export.ok" ] && [ ! -f "$R/$cell.export.job" ]; then
      j=$(submit -A CCR24067 -p debug -N 4 -t 00:25:00 -J "cf2e57_export_$cell" -o "$R/logs/export-$cell.%j.log" --export=ALL \
          "$MOE/horizon_snowball_export.sbatch")
      [ -n "$j" ]; echo "$j" > "$R/$cell.export.job"; say "export $cell job$j"
    fi
  done
done
for cell in fdv5 fdv7 fdr5 fdr7; do
  if [ ! -f "$R/$cell.export.ok" ]; then
    j=$(cat "$R/$cell.export.job")
    wait_ok "$j" "$R/logs/export-$cell.$j.log" EXPORT_CHECK_OK
    touch "$R/$cell.export.ok"; say "export $cell OK"
  fi
done
export MODELS REPS="1 2 3" GROUPED=1 MAXJOBS=18 SERVE_TIME=03:00:00
export HARBOR_SRC=$HOME/snowball/harbor-p0924/src HARBOR_SHA=761fb516 POLICY_FILE=tb2_marin_policy_0924_65k16k.yaml
export DAY=cf2e57_20261022 SETS="tb21 swe_s0 swe_s1 tblite_s0 tblite_s1"
if [ ! -f "$R/eval.ok" ]; then
  while :; do
    ds=$(python3 "$REPLAY_OTA/data/r2egym/horizon/sft/daytona_started.py" 2>/dev/null || true)
    [[ "$ds" =~ ^[0-9]+$ ]] && [ $((ds + 960)) -le 2000 ] && break
    say "Waiting for sandbox admission (started=$ds; need 960; limit 2000)"; sleep 60
  done
  say "EVAL_START: 12 groups, 960 concurrent trials, 3456 attempts"
  bash "$REPLAY_OTA/data/r2egym/horizon/sft/eval_sft.sh" > "$R/logs/eval.log" 2>&1
  touch "$R/eval.ok"
fi
python3 "$REPLAY_OTA/data/r2egym/horizon/sft/eval_readout.py" --tags cv5 cr5 cv7 cr7 \
  --pairs-with cv5 cr5 cv7 cr7 --out "$R/readout.json" > "$R/readout.txt"
say "READOUT_WRITTEN: audit missing scores before comparison"
