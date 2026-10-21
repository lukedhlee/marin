#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# One-shot replay of cfvanilla/H8, isolated from existing Horizon jobs.
set -euo pipefail
: "${FP32_REPLAY_ROOT:?}"
: "${MARIN_ROOT:?}"
: "${REPLAY_OTA:?frozen OTA checkout at 1c40b0bfff98ffae6ff70f5a0496b1fc3db3eebb}"
R=$FP32_REPLAY_ROOT
MOE=$MARIN_ROOT/experiments/june_tpu_67b_a2b/moe
OLD=/scratch/11584/$USER/snowball-sft
BASE=/scratch/11584/$USER/models/grug-datakit-sft-20260921
export MARIN_PYTHON=$HOME/snowball/envs/marin-grug-sft/bin/python
mkdir -p "$R/bin" "$R/logs" "$R/training"
ln -sfn "$MOE/replay_sbatch.py" "$R/bin/sbatch"
export PATH="$R/bin:$PATH"
export SNOWBALL_SCRATCH=$R SNOWBALL_INIT=$OLD/experiments/snowball-base-inits/init-dk0921-step0
export SNOWBALL_TOKENIZER=$BASE SNOWBALL_SEED=0 SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16
export SNOWBALL_NODES=4 SNOWBALL_DEVICES=16 SNOWBALL_LR=3e-4
export SNOWBALL_SAVE_INTERVAL_MIN=100000 SNOWBALL_KEEP_PER_EPOCH=1
export SNOWBALL_PROBE_MEMORY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset SNOWBALL_PROBE_STEPS SNOWBALL_KEEP_EVERY SNOWBALL_RESUME_STEP SNOWBALL_TAIL_FRACTION
unset SNOWBALL_TAIL_REF SNOWBALL_TAIL_RAMP SNOWBALL_TAIL_SCORE_OUT SNOWBALL_XLA_EXTRA
say() { echo "[$(date -u +%FT%TZ)] $*"; }
wait_job() {
  local j=$1 log=$2 marker=$3
  while squeue -h -j "$j" -t PD,R,CF,S,RQ,RS | grep -q .; do sleep 30; done
  grep -q "$marker" "$log" || { say "FAILED job$j: $log"; return 1; }
}
submit() {
  local output rc
  while :; do
    set +e
    output=$(sbatch "$@" 2>&1); rc=$?
    set -e
    if [ "$rc" = 75 ]; then say "$output" >&2; sleep 60; continue; fi
    if [ "$rc" != 0 ]; then say "$output" >&2; return "$rc"; fi
    echo "$output" | sed -nE 's/Submitted batch job ([0-9]+).*/\1/p' | tail -1
    return
  done
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
# The smoke uses the real cache/init/geometry and the full-run LR schedule.
SMOKE_JOB=$(python3 -c 'import json,os; print(json.load(open(os.environ["FP32_REPLAY_ROOT"]+"/smoke_submission.json"))["job_id"])')
wait_job "$SMOKE_JOB" "$R/logs/smoke.$SMOKE_JOB.log" 'GUARDED_RUN_EXIT rc=0'
check_dtype "$R/training/smoke-cfvanilla/checkpoints/step-2"
say "SMOKE_GATE_PASS"
select_arm() {
  local arm=$1
  case "$arm" in
    cfvanilla) epoch=221; warmup=33; wall=02:15:00;;
    allkimi) epoch=401; warmup=60; wall=03:30:00;;
  esac
  export SNOWBALL_STAGE=relay_$arm SNOWBALL_STEPS=$((3 * epoch)) SNOWBALL_EPOCH_STEPS=$epoch
  export SNOWBALL_SCHEDULE_STEPS=$SNOWBALL_STEPS SNOWBALL_WARMUP=$warmup
  export SNOWBALL_CACHE=$OLD/experiments/snowball-relay-sft/cache-relay_$arm-v1
  export SNOWBALL_OUTPUT=$R/training/$arm SNOWBALL_RUN_ID=fp32-${arm}-0921-20261020
}
# Submit both independent training arms, adopting a recorded job on restart.
# Each submission still checks all queued/running requested SUs.
for arm in cfvanilla allkimi; do
  select_arm "$arm"
  if [ ! -f "$R/$arm.train.job" ] && [ ! -f "$R/$arm.train.ok" ]; then
    [ ! -e "$SNOWBALL_OUTPUT" ] || { say "Refusing to overwrite $SNOWBALL_OUTPUT"; exit 1; }
    j=$(submit -N 4 -t "$wall" -J "fp32_$arm" -o "$R/logs/$arm.%j.log" --export=ALL "$MOE/horizon_snowball_guarded.sbatch")
    [ -n "$j" ]; echo "$j" > "$R/$arm.train.job"
    say "TRAIN_SUBMITTED $arm job$j steps$SNOWBALL_STEPS"
  fi
done
for arm in cfvanilla allkimi; do
  select_arm "$arm"
  if [ ! -f "$R/$arm.train.ok" ]; then
    j=$(cat "$R/$arm.train.job")
    say "TRAIN_ADOPTED $arm job$j"
    wait_job "$j" "$R/logs/$arm.$j.log" 'GUARDED_RUN_EXIT rc=0'
    for step in "$epoch" "$((2 * epoch))" "$((3 * epoch))"; do
      check_dtype "$SNOWBALL_OUTPUT/checkpoints/step-$step"
    done
    touch "$R/$arm.train.ok"
  fi
  for step in "$epoch" "$((2 * epoch))" "$((3 * epoch))"; do
    export SNOWBALL_EXPORT_CHECKPOINT=$SNOWBALL_OUTPUT/checkpoints/step-$step
    export SNOWBALL_EXPORT_OUTPUT=$SNOWBALL_OUTPUT/export-step$step-hf-bf16
    export SNOWBALL_EXPORT_TOKENIZER=$BASE SNOWBALL_EXPORT_BASE=$BASE
    if [ ! -f "$R/$arm.export$step.ok" ]; then
      j=$(submit -N 4 -t 00:25:00 -J "fp32_export" -o "$R/logs/export-$arm-$step.%j.log" --export=ALL "$MOE/horizon_snowball_export.sbatch")
      [ -n "$j" ]; echo "$j" > "$R/$arm.export$step.job"
      say "EXPORT_SUBMITTED $arm step$step job$j"
      wait_job "$j" "$R/logs/export-$arm-$step.$j.log" 'EXPORT_CHECK_OK'
      touch "$R/$arm.export$step.ok"
    fi
  done
  say "ARM_DONE $arm"
done
say "TRAINING_DONE; starting paired original/fixed evaluation"
# Keep two units at most. Every unit is one (model,rep), five nodes. Repetition
# order alternates old/fixed to limit temporal load confounding.
export HARBOR_SRC=$HOME/snowball/harbor-p0924/src HARBOR_SHA=761fb516
export POLICY_FILE=tb2_marin_policy_0924_65k16k.yaml
export SERVE_TIME=03:00:00 MAXJOBS=20 GROUPED=1 DAY=fp32_20261020
oldmodels=$OLD/experiments/snowball-relay-sft
pids=()
for rep in 1 2 3; do
  for entry in \
    "bf16van=$oldmodels/relay_cfvanilla/lr3e-4-sched3/export-step663-hf-bf16" \
    "fp32van=$R/training/cfvanilla/export-step663-hf-bf16" \
    "bf16h8=$oldmodels/relay_allkimi/lr3e-4-sched3/export-step1203-hf-bf16" \
    "fp32h8=$R/training/allkimi/export-step1203-hf-bf16" \
    "basefp=$BASE"; do
    tag=${entry%%=*}
    # Check existing eval-org load through the same probe as the original dispatcher.
    while :; do
      ds=$(python3 "$REPLAY_OTA/data/r2egym/horizon/sft/daytona_started.py")
      [ $((ds + 80)) -le 850 ] && break
      say "Waiting for Daytona room: $ds started"; sleep 60
    done
    MODELS=$entry REPS=$rep bash "$REPLAY_OTA/data/r2egym/horizon/sft/eval_sft.sh" > "$R/logs/eval-$tag-r$rep.log" 2>&1 &
    pids+=("$!")
    if [ "${#pids[@]}" = 2 ]; then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  pids=()
done
say "EVALUATION_JOBS_DONE; readouts require coverage/error audit before interpreting scores"
python3 "$REPLAY_OTA/data/r2egym/horizon/sft/eval_readout.py" \
  --tags fp32van bf16van fp32h8 bf16h8 basefp --day "$DAY" \
  --out "$R/paired_readout.json" > "$R/paired_readout.txt"
python3 - "$R" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);d=json.loads((r/'paired_readout.json').read_text())
checks=[]
for tag in ['fp32van','bf16van','fp32h8','bf16h8','basefp']:
    for family,expected in [('tb21',88),('swe',100),('tblite',100)]:
        result=d.get(f'{tag}/{family}',{})
        reps=result.get('reps',{})
        checks.append({'tag':tag,'family':family,'repetitions':len(reps),'expected_tasks_per_rep':expected,
                       'scored_per_rep':{k:v.get('n',0) for k,v in reps.items()},
                       'passed':len(reps)==3 and all(v.get('n',0)>=.9*expected for v in reps.values())})
x={'passed':all(x['passed'] for x in checks),'checks':checks,
   'note':'Do not interpret a missing/low-coverage comparison as a model effect; inspect errors and apply symmetric infrastructure recovery first.'}
(r/'evaluation_coverage.json').write_text(json.dumps(x,indent=2)+'\n')
print('EVALUATION_COVERAGE',x['passed'])
PY
say "REPLAY_FINISHED; paired_readout.json and evaluation_coverage.json written"
