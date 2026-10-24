#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Mixed CalibForge SFT: an approximately token-balanced budget and the full Vanilla + Relay corpus.
set -euo pipefail
unset LD_PRELOAD
: "${MARIN_ROOT:?}"
: "${FP32_REPLAY_ROOT:?}"
: "${REPLAY_OTA:?}"
R=$FP32_REPLAY_ROOT
F=/scratch/11584/$USER/experiments/relay-fixed-design-20261021/final
MOE=$MARIN_ROOT/experiments/june_tpu_67b_a2b/moe
BASE=/scratch/11584/$USER/models/grug-datakit-sft-20260921
export MARIN_PYTHON=${MARIN_PYTHON:-$HOME/snowball/envs/marin-grug-sft/bin/python}
export PYTHONPATH="$MARIN_ROOT:$MARIN_ROOT/lib/levanter/src:$MARIN_ROOT/lib/haliax/src:$MARIN_ROOT/lib/marin/src:$MARIN_ROOT/lib/fray/src:$MARIN_ROOT/lib/zephyr/src:$MARIN_ROOT/lib/iris/src:$MARIN_ROOT/lib/rigging/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CC=gcc CXX=g++
cd "$MARIN_ROOT"
case "${1:-driver}" in
  data)
    : "${SLURM_JOB_ID:?data preparation must run in a compute allocation}"
    export JAX_PLATFORMS=cpu
    export LD_PRELOAD=$(gcc -print-file-name=libstdc++.so.6)
    exec "$MARIN_PYTHON" "$MOE/prepare_cf_mixed.py" --source-root "$F" --output-root "$R" \
      --source-manifest "$F/../fixed_design_report.json"
    ;;
  verify)
    : "${SLURM_JOB_ID:?cache audit must run in a compute allocation}"
    : "${MIXED_ARM:?}"
    export JAX_PLATFORMS=cpu
    export LD_PRELOAD=$(gcc -print-file-name=libstdc++.so.6)
    "$MARIN_PYTHON" "$MOE/verify_cf_mixed_cache.py" "relay_$MIXED_ARM" \
      "$R/data/$MIXED_ARM/rows.jsonl" "$R/data/$MIXED_ARM" "$R/cache-relay_$MIXED_ARM-v1" "$BASE"
    ;;
  driver) ;;
  *) echo "usage: $0 [driver|data|verify]" >&2; exit 2 ;;
esac
[ "${1:-driver}" = driver ] || exit 0
# This metadata-only branch imports no training packages, touches no run files, and submits nothing.
if [ "${DRY_RUN:-0}" = 1 ]; then
  python3 - "$MARIN_ROOT" "$R" "$REPLAY_OTA" <<'PY'
import json, sys
from pathlib import Path
m, r, ota = map(Path, sys.argv[1:])
moe = m / 'experiments/june_tpu_67b_a2b/moe'
for filename in ['model.py', 'vista_snowball_chat.py', 'prepare_cf_mixed.py', 'verify_cf_mixed_cache.py',
                 'replay_sbatch.py', 'horizon_r2egym_prep.sbatch', 'horizon_snowball_guarded.sbatch',
                 'horizon_snowball_export.sbatch']:
    assert (moe / filename).is_file(), filename
assert '_DEFAULT_EP_CAPACITY_FACTOR = 2.0' in (moe / 'model.py').read_text()
for arm in ['cfmix50', 'cfconcat']:
    assert f'"{arm}"' in (moe / 'vista_snowball_chat.py').read_text(), arm
assert 'dataset_revision="cfmix20261023"' in (moe / 'vista_snowball_chat.py').read_text()
assert 'SNOWBALL_CHAT_MP = "params=float32,compute=bfloat16,output=bfloat16"' in (moe / 'snowball_chat_recipe.py').read_text()
train_source = (moe / 'train.py').read_text()
assert 'params = mp.cast_to_param(loaded["params"])' in train_source
assert '"opt_state": optimizer.init(params)' in train_source
assert 'leaf.dtype != trainer.mp.param_dtype' in train_source
print(json.dumps(dict(root=str(r), arms=['cfmix50', 'cfconcat'], source='fixed-design fdv/fdr',
                     nominal_budget='cfmix50: half of the original Vanilla supervised-token mass per source, '
                                    'rounded to whole rows; cfconcat: all rows from both sources',
                     actual_loader_exposure='not asserted equal to the nominal corpus budget',
                     passes=3, batch=16, sequence_length=65536, nodes_per_training=4, seed=0, lr=3e-4,
                     master_dtype='float32', capacity_factor=2.0, smoke_steps=6,
                     smoke_max_capacity_overflow=0.02, max_own_reserved_nodes=100,
                     eval='separate repaired-harness stage after exports'), indent=2))
print('DRY_RUN_OK: no jobs submitted')
PY
  exit 0
fi
mkdir -p "$R/bin" "$R/logs" "$R/training"
exec 9>"$R/driver.lock"
flock -n 9 || { echo "Another driver owns $R" >&2; exit 1; }
trap 'echo "FAILED at line $LINENO; inspect driver.log and phase logs" > "$R/STATUS"' ERR
[ -z "$(git status --porcelain --untracked-files=no)" ]
grep -q '^_DEFAULT_EP_CAPACITY_FACTOR = 2.0$' "$MOE/model.py"
grep -Fq 'SNOWBALL_CHAT_MP = "params=float32,compute=bfloat16,output=bfloat16"' "$MOE/snowball_chat_recipe.py"
grep -Fq 'params = mp.cast_to_param(loaded["params"])' "$MOE/train.py"
grep -Fq '"opt_state": optimizer.init(params)' "$MOE/train.py"
grep -Fq 'leaf.dtype != trainer.mp.param_dtype' "$MOE/train.py"
ln -sfn "$MOE/replay_sbatch.py" "$R/bin/sbatch"
export PATH="$R/bin:$PATH"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$R/driver.log" > "$R/STATUS"; }
submit() {
  local requested=$1 output rc nodes; shift
  exec 8>"$R/node-admission.lock"
  while :; do
    flock 8
    nodes=$(squeue -u "$USER" -h -o %D | awk '{n+=$1} END {print n+0}') || return
    if [ $((nodes + requested)) -gt 100 ]; then flock -u 8; sleep 30; continue; fi
    set +e; output=$(sbatch "$@" 2>&1); rc=$?; set -e
    flock -u 8
    if [ "$rc" = 75 ]; then echo "$output" >&2; sleep 30; continue; fi
    [ "$rc" = 0 ] || { echo "$output" >&2; return "$rc"; }
    echo "$output" | sed -nE 's/Submitted batch job ([0-9]+).*/\1/p' | tail -1
    return
  done
}
phase() {
  local name=$1 marker=$2 nodes=$3 j queue accounting; shift 3
  [ -f "$R/$name.ok" ] && return
  if [ ! -f "$R/$name.job" ]; then
    j=$(submit "$nodes" -A CCR24067 -p debug -N "$nodes" -o "$R/logs/$name.%j.log" "$@")
    [ -n "$j" ]; echo "$j" > "$R/$name.job"
    say "$name submitted job$j"
  fi
  j=$(cat "$R/$name.job")
  while :; do
    queue=$(squeue -u "$USER" -h -o '%i|%T' | awk -F'|' -v job="$j" '$1 == job {print $2}') || return
    [ -n "$queue" ] || break
    if [ "$queue" = COMPLETING ]; then
      accounting=$(sacct -j "$j" -n -P --format=JobIDRaw,State,ExitCode | \
        awk -F'|' -v job="$j" 'index($1, ".") == 0 && $1 == job {print $2 "|" $3}') || return
      [ "$accounting" != 'COMPLETED|0:0' ] || break
    fi
    sleep 30
  done
  grep -q "$marker" "$R/logs/$name.$j.log"
  touch "$R/$name.ok"
}
check_dtype() {
  "$MARIN_PYTHON" - "$1" <<'PY'
import json, sys, tensorstore as ts
from pathlib import Path
p = Path(sys.argv[1])
assert (p / 'metadata.json').is_file(), p
params = [a for a in json.loads((p / 'manifest.json').read_text())['arrays'] if a['path'].startswith('params/')]
assert params and all(a['dtype'] == 'float32' for a in params), p
array = ts.open({'driver': 'zarr3', 'kvstore': {'driver': 'ocdbt',
                'base': {'driver': 'file', 'path': str(p) + '/'}, 'path': 'params/token_embed'}},
                open=True, read=True, write=False).result()
assert array.dtype.name == 'float32', array.dtype
print('FP32_CHECKPOINT_OK', p, flush=True)
PY
}
select_arm() {
  local arm=$1
  epoch=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["pack_epoch_steps"])' "$R/data/$arm/pack_epoch_steps.json")
  [ "$epoch" -gt 2 ]
  steps=$((3 * epoch)); warmup=$(((steps * 5 + 50) / 100)); [ "$warmup" -ge 2 ] || warmup=2
  export SNOWBALL_SCRATCH=$R SNOWBALL_INIT=/scratch/11584/$USER/snowball-sft/experiments/snowball-base-inits/init-dk0921-step0
  export SNOWBALL_TOKENIZER=$BASE SNOWBALL_SEED=0 SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16
  export SNOWBALL_NODES=4 SNOWBALL_DEVICES=16 SNOWBALL_LR=3e-4 SNOWBALL_STAGE=relay_$arm
  export SNOWBALL_FREEZE_ROUTER_BIAS=1
  export SNOWBALL_STEPS=$steps SNOWBALL_SCHEDULE_STEPS=$steps SNOWBALL_EPOCH_STEPS=$epoch SNOWBALL_WARMUP=$warmup
  export SNOWBALL_CACHE=$R/cache-relay_$arm-v1 SNOWBALL_PARQUET_LIST=$R/data/$arm/parquet.list
  export SNOWBALL_DATASET_ID=private/relay-calibforge-sft-$arm SNOWBALL_DATASET_REVISION=cfmix20261023
  export SNOWBALL_OUTPUT=$R/training/$arm SNOWBALL_RUN_ID=fp32-cap2-$arm-0921-20261023
  export SNOWBALL_SAVE_INTERVAL_MIN=100000 SNOWBALL_KEEP_PER_EPOCH=1 SNOWBALL_PROBE_MEMORY=0
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
  unset SNOWBALL_PROBE_STEPS SNOWBALL_KEEP_EVERY SNOWBALL_RESUME_STEP SNOWBALL_TAIL_FRACTION
  unset SNOWBALL_TAIL_REF SNOWBALL_TAIL_RAMP SNOWBALL_TAIL_SCORE_OUT SNOWBALL_XLA_EXTRA
}
smoke_gate() {
  local arm=$1
  check_dtype "$R/training/smoke-$arm/checkpoints/step-6"
  "$MARIN_PYTHON" - "$R" "$MARIN_ROOT" "$arm" <<'PY'
import json, math, sys
from pathlib import Path
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore
r, m, arm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
files = list((m / 'wandb').glob(f'offline-run-*-fp32-cap2-{arm}-smoke-20261023/*.wandb'))
assert len(files) == 1, files
store = DataStore(); store.open_for_scan(str(files[0]))
drops, losses = [], []
while (raw := store.scan_data()) is not None:
    record = wandb_internal_pb2.Record(); record.ParseFromString(raw)
    for item in record.history.item:
        key = item.key or '/'.join(item.nested_key)
        if key == 'train/router/capacity_overflow_rate_mean':
            drops.append(float(json.loads(item.value_json)))
        elif key == 'train/loss':
            losses.append(float(json.loads(item.value_json)))
assert len(drops) >= 6 and all(math.isfinite(x) and 0 <= x < 0.02 for x in drops), drops
assert len(losses) >= 6 and all(math.isfinite(x) for x in losses), losses
(r / f'{arm}.smoke_gate.json').write_text(json.dumps(dict(capacity=2.0, master_dtype='float32',
    max_allowed_overflow=0.02, drops=drops, losses=losses, wandb_file=str(files[0])), indent=2) + '\n')
print('CF_MIXED_SMOKE_OK', arm, flush=True)
PY
}
say "START mixed SFT commit=$(git rev-parse HEAD)"
printf -v data_command 'bash %q data' "$MOE/priority_cf_mixed.sh"
phase data CF_MIXED_DATA_OK 1 -t 01:00:00 --ntasks-per-node=1 --cpus-per-task=16 \
  -J cfmix_data --export=ALL --wrap="$data_command"
[ -f "$R/prepared.json" ]
arm_pipeline() {
  local arm=$1 command j
  select_arm "$arm"
  phase "$arm.cache" PREP_DONE 1 -t 01:00:00 -J "${arm}_cache" \
    --export=ALL,SNOWBALL_ENV="$(dirname "$(dirname "$MARIN_PYTHON")")",PREP_STEPS=cache "$MOE/horizon_r2egym_prep.sbatch"
  export MIXED_ARM=$arm
  printf -v command 'bash %q verify' "$MOE/priority_cf_mixed.sh"
  phase "$arm.verify" '"verdict": "EQUAL"' 1 -t 01:00:00 --ntasks-per-node=1 --cpus-per-task=16 \
    -J "${arm}_verify" --export=ALL --wrap="$command"
  export SNOWBALL_STEPS=6 SNOWBALL_KEEP_EVERY=6
  unset SNOWBALL_KEEP_PER_EPOCH
  export SNOWBALL_OUTPUT=$R/training/smoke-$arm SNOWBALL_RUN_ID=fp32-cap2-$arm-smoke-20261023
  phase "$arm.smoke" 'GUARDED_RUN_EXIT rc=0' 4 -t 00:40:00 -J "${arm}_smoke" \
    --export=ALL "$MOE/horizon_snowball_guarded.sbatch"
  smoke_gate "$arm" > "$R/logs/$arm.smoke_gate.log" 2>&1
  select_arm "$arm"
  # Training and export always have separate allocations; no overlapping JAX program is attached to training.
  phase "$arm.train" 'GUARDED_RUN_EXIT rc=0' 4 -t 04:30:00 -J "${arm}_train" \
    --export=ALL "$MOE/horizon_snowball_guarded.sbatch"
  for j in "$epoch" "$((2 * epoch))" "$steps"; do
    check_dtype "$SNOWBALL_OUTPUT/checkpoints/step-$j"
  done
  export SNOWBALL_EXPORT_CHECKPOINT=$SNOWBALL_OUTPUT/checkpoints/step-$steps
  export SNOWBALL_EXPORT_OUTPUT=$SNOWBALL_OUTPUT/export-step$steps-hf-bf16
  export SNOWBALL_EXPORT_TOKENIZER=$BASE SNOWBALL_EXPORT_BASE=$BASE
  phase "$arm.export" EXPORT_CHECK_OK 4 -t 00:45:00 -J "${arm}_export" \
    --export=ALL "$MOE/horizon_snowball_export.sbatch"
  printf '%s\n' "$SNOWBALL_EXPORT_OUTPUT" > "$R/$arm.export.path"
  say "EXPORT_READY $arm $SNOWBALL_EXPORT_OUTPUT"
}
pids=()
for arm in cfmix50 cfconcat; do
  ( arm_pipeline "$arm" ) > "$R/logs/pipeline-$arm.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" = 0 ]
say "EXPORTS_READY_FOR_EVAL cfmix50=$(cat "$R/cfmix50.export.path") cfconcat=$(cat "$R/cfconcat.export.path")"
