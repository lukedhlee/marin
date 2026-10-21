#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# One-shot vanilla CalibForge + Kimi: prepare -> cache -> capacity2 smoke -> train -> export -> evaluate.
set -euo pipefail
unset LD_PRELOAD
: "${MARIN_ROOT:?}"
: "${FP32_REPLAY_ROOT:?}"
: "${REPLAY_OTA:?}"
R=$FP32_REPLAY_ROOT
MOE=$MARIN_ROOT/experiments/june_tpu_67b_a2b/moe
BASE=/scratch/11584/$USER/models/grug-datakit-sft-20260921
export MARIN_PYTHON=$HOME/snowball/envs/marin-grug-sft/bin/python
export PYTHONPATH="$MARIN_ROOT:$MARIN_ROOT/lib/levanter/src:$MARIN_ROOT/lib/haliax/src:$MARIN_ROOT/lib/marin/src:$MARIN_ROOT/lib/fray/src:$MARIN_ROOT/lib/zephyr/src:$MARIN_ROOT/lib/iris/src:$MARIN_ROOT/lib/rigging/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$MARIN_ROOT"
if [ "${1:-driver}" = data ]; then
  export JAX_PLATFORMS=cpu
  export LD_PRELOAD=$(gcc -print-file-name=libstdc++.so.6)
  exec "$MARIN_PYTHON" "$MOE/prepare_priority_vanilla_kimi.py"
fi
mkdir -p "$R/bin" "$R/logs" "$R/training"
exec 9>"$R/driver.lock"
flock -n 9 || { echo "Another priority driver owns $R"; exit 1; }
trap 'echo "FAILED at line $LINENO; inspect driver.log and phase logs" > "$R/STATUS"' ERR
[ -z "$(git status --porcelain --untracked-files=no)" ]
grep -q '^_DEFAULT_EP_CAPACITY_FACTOR = 2.0$' "$MOE/model.py"
ln -sfn "$MOE/replay_sbatch.py" "$R/bin/sbatch"
export PATH="$R/bin:$PATH"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee "$R/STATUS"; }
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
phase() {
  local name=$1 marker=$2 j; shift 2
  [ -f "$R/$name.ok" ] && return
  if [ ! -f "$R/$name.job" ]; then
    j=$(submit -o "$R/logs/$name.%j.log" "$@")
    [ -n "$j" ]; echo "$j" > "$R/$name.job"
    say "$name submitted job$j"
  fi
  j=$(cat "$R/$name.job")
  while squeue -h -j "$j" -t PD,R,CF,S,RQ,RS | grep -q .; do sleep 30; done
  grep -q "$marker" "$R/logs/$name.$j.log"
  touch "$R/$name.ok"
}
say "START vanilla CalibForge+Kimi FP32 capacity2 commit=$(git rev-parse HEAD)"
if [ ! -f "$R/prepared.json" ]; then
  phase data PRIORITY_DATA_OK -A CCR24067 -p debug -N 1 -t 01:00:00 --ntasks-per-node=1 --cpus-per-task=16 \
    -J vkimicf2_data --export=ALL --wrap="bash '$MOE/priority_vanilla_kimi.sh' data"
fi
epoch=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["pack_epoch_steps"])' "$R/pack_epoch_steps.json")
steps=$((3 * epoch)); warmup=$(((steps * 5 + 50) / 100)); [ "$warmup" -ge 2 ] || warmup=2
export SNOWBALL_SCRATCH=$R SNOWBALL_INIT=/scratch/11584/$USER/snowball-sft/experiments/snowball-base-inits/init-dk0921-step0
export SNOWBALL_TOKENIZER=$BASE SNOWBALL_SEED=0 SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16
export SNOWBALL_NODES=4 SNOWBALL_DEVICES=16 SNOWBALL_LR=3e-4 SNOWBALL_STAGE=relay_fdvk
export SNOWBALL_SCHEDULE_STEPS=$steps SNOWBALL_EPOCH_STEPS=$epoch SNOWBALL_WARMUP=$warmup
export SNOWBALL_CACHE=$R/cache-relay_fdvk-v1 SNOWBALL_PARQUET_LIST=$R/data/parquet.list
export SNOWBALL_DATASET_ID=private/relay-calibforge-sft-fdvk SNOWBALL_DATASET_REVISION=fdvkcf2_20261021
export SNOWBALL_SAVE_INTERVAL_MIN=100000 SNOWBALL_KEEP_PER_EPOCH=1 SNOWBALL_PROBE_MEMORY=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
unset SNOWBALL_PROBE_STEPS SNOWBALL_KEEP_EVERY SNOWBALL_RESUME_STEP SNOWBALL_TAIL_FRACTION
unset SNOWBALL_TAIL_REF SNOWBALL_TAIL_RAMP SNOWBALL_TAIL_SCORE_OUT SNOWBALL_XLA_EXTRA
phase cache PREP_DONE -N 1 -t 01:00:00 -J vkimicf2_cache \
  --export=ALL,SNOWBALL_ENV="$(dirname "$(dirname "$MARIN_PYTHON")")",PREP_STEPS=cache "$MOE/horizon_r2egym_prep.sbatch"
export SNOWBALL_STEPS=6 SNOWBALL_OUTPUT=$R/training/smoke SNOWBALL_RUN_ID=fp32-vkimicf2-smoke-20261021
phase smoke 'GUARDED_RUN_EXIT rc=0' -N 4 -t 00:40:00 -J vkimicf2_smoke --export=ALL "$MOE/horizon_snowball_guarded.sbatch"
if [ ! -f "$R/smoke_gate.json" ]; then
  JAX_PLATFORMS=cpu "$MARIN_PYTHON" - "$R" "$MARIN_ROOT" <<'PY'
import glob,json,math,sys
from pathlib import Path
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2
r,m=map(Path,sys.argv[1:]);p=r/'training/smoke/checkpoints/step-6'
arrays=json.load(open(p/'manifest.json'))['arrays']
assert all(a['dtype']=='float32' for a in arrays if a['path'].startswith('params/'))
files=glob.glob(str(m/'wandb/offline-run-*-fp32-vkimicf2-smoke-20261021/*.wandb'));assert len(files)==1,files
d=DataStore();d.open_for_scan(files[0]);drops=[];losses=[]
while True:
    raw=d.scan_data()
    if raw is None:break
    rec=wandb_internal_pb2.Record();rec.ParseFromString(raw)
    for item in rec.history.item:
        key=item.key or '/'.join(item.nested_key)
        if key.endswith('capacity_overflow_rate_mean'):drops.append(float(json.loads(item.value_json)))
        if key=='train/cross_entropy_loss':losses.append(float(json.loads(item.value_json)))
assert len(drops)>=6 and max(drops)<0.01,drops
assert losses and all(math.isfinite(x) for x in losses),losses
(r/'smoke_gate.json').write_text(json.dumps(dict(capacity=2.0,master_dtype='float32',drops=drops,losses=losses),indent=2)+'\n')
print('CAPACITY2_SMOKE_OK',drops,losses,flush=True)
PY
fi
export SNOWBALL_STEPS=$steps SNOWBALL_OUTPUT=$R/training/vanilla_kimi SNOWBALL_RUN_ID=fp32-vkimicf2-0921-20261021
# Allow startup, three checkpoint saves, and slower capacity2 dispatch. No old run is resumed.
phase train 'GUARDED_RUN_EXIT rc=0' -N 4 -t 04:00:00 -J fp32_vkimicf2 --export=ALL "$MOE/horizon_snowball_guarded.sbatch"
python3 - "$SNOWBALL_OUTPUT/checkpoints/step-$steps/manifest.json" <<'PY'
import json,sys
a=[a for a in json.load(open(sys.argv[1]))['arrays'] if a['path'].startswith('params/')]
assert a and all(x['dtype']=='float32' for x in a)
PY
export SNOWBALL_EXPORT_CHECKPOINT=$SNOWBALL_OUTPUT/checkpoints/step-$steps
export SNOWBALL_EXPORT_OUTPUT=$SNOWBALL_OUTPUT/export-step$steps-hf-bf16
export SNOWBALL_EXPORT_TOKENIZER=$BASE SNOWBALL_EXPORT_BASE=$BASE
phase export EXPORT_CHECK_OK -N 4 -t 00:25:00 -J vkimicf2_export --export=ALL "$MOE/horizon_snowball_export.sbatch"
export MODELS="vkimicf2=$SNOWBALL_EXPORT_OUTPUT" REPS="1 2" GROUPED=1 MAXJOBS=20 SERVE_TIME=03:00:00
export HARBOR_SRC=$HOME/snowball/harbor-p0924/src HARBOR_SHA=761fb516 POLICY_FILE=tb2_marin_policy_0924_65k16k.yaml
# Get SWE evidence first, then the terminal sets. Separate dates prevent group-marker collisions.
for family in swe terminal; do
  [ -f "$R/eval.$family.ok" ] && continue
  while :; do
    ds=$(python3 "$REPLAY_OTA/data/r2egym/horizon/sft/daytona_started.py" 2>/dev/null || true)
    [[ "$ds" =~ ^[0-9]+$ ]] && [ $((ds + 96)) -le 850 ] && break
    say "Waiting for Daytona room before $family eval"; sleep 60
  done
  export DAY=vkimicf2_${family}_20261021
  if [ "$family" = swe ]; then export SETS="swe_s0 swe_s1"; else export SETS="tb21 tblite_s0 tblite_s1"; fi
  say "EVAL_START $family"
  bash "$REPLAY_OTA/data/r2egym/horizon/sft/eval_sft.sh" > "$R/logs/eval-$family.log" 2>&1
  touch "$R/eval.$family.ok"
done
python3 "$REPLAY_OTA/data/r2egym/horizon/sft/eval_readout.py" --tags vkimicf2 \
  --day 'vkimicf2_*_20261021' --out "$R/readout.json" > "$R/readout.txt"
say "READOUT_WRITTEN: audit coverage and errors before comparing scores"
