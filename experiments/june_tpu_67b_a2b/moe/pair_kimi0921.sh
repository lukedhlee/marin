#!/bin/bash
# pair_kimi0921.sh — one side of the Horizon <-> Jupiter SFT port pair (HORIZON_JUPITER_PAIR.md at the repo root).
# The CURRENT Snowball SFT recipe on public data, identical on both clusters: Grug Datakit 09-21 base imported with
# pending_qb_betas = -router_bias (router bias frozen at 09-21's), 09-21 training template, datakit think rows,
# 16 x 65,536 tokens per step on 4 nodes (16 one-GPU ranks), lr 3e-4, warmup 2 steps, one cosine over 30 steps, seed 0.
#
#   CLUSTER=horizon|jupiter [REPLICA=r1] bash pair_kimi0921.sh      (login node, inside tmux; it only submits and polls)
#
# Steps, each skipped when its artifact exists: data (HF download + sha256 check) -> prep (cache, 1 node) and import
# (09-21 -> native init with its sidecar, 4 CPU nodes; Jupiter reuses the Bespoke/relay init) -> run (4 nodes, 30
# steps) -> W&B sync of the offline run + a per-step loss TSV. Cost per side: ~0.3 + 0.3 + ~1.5 node-h.
set -uo pipefail
CLUSTER=${CLUSTER:?horizon | jupiter}; REPLICA=${REPLICA:-r1}
STAGE=kimi0921_pair
HF_DATA=lukeleeai/snowball-kimi0921-pair
HF_DATA_REV=2883a32db3232284bae7a65130a30ded0b696519
PARQUET_SHA=7683b2ff5177159a099939ba6ea9578130d97f7175e401048da5738c18d1a094
WANDB_ENTITY_PAIR=lukedhlee-marin; WANDB_PROJECT_PAIR=horizon-jupiter-sft-pair
MOE=$(cd "$(dirname "$0")" && pwd); MARIN=$(cd "$MOE/../../.." && pwd)
case $CLUSTER in
  horizon)
    S=${SNOWBALL_SCRATCH:-/scratch/11584/$USER/snowball-sft}
    PYM=${MARIN_PYTHON:-$HOME/snowball/envs/marin-grug-sft/bin/python}
    BASE=${BASE_0921:-/scratch/11584/$USER/models/grug-datakit-sft-20260921}
    INIT=${SNOWBALL_INIT:-$S/experiments/snowball-base-inits/init-dk0921-step0}
    HF=${HF_CLI:-$HOME/snowball/envs/snowball/bin/hf}; WANDB=${WANDB_CLI:-$HOME/snowball/envs/snowball/bin/wandb}
    KEYS=${KEYS:-$HOME/.config/otagent/secrets.env}
    P=horizon; ACCT=();;
  jupiter)
    S=${SNOWBALL_SCRATCH:-/e/data1/mmlaion/$USER/snowball-sft}
    PYM=${MARIN_PYTHON:-/e/project1/transfernetx/$USER/code/envs/marin-grug-sft/bin/python}
    BASE=${BASE_0921:-/e/data1/mmlaion/$USER/models/grug-datakit-sft-20260921}
    INIT=${SNOWBALL_INIT:-$S/experiments/snowball-base-inits/init-dk0921-step0}
    HF=${HF_CLI:-/e/project1/transfernetx/$USER/code/envs/snowball-v2/bin/hf}; WANDB=${WANDB_CLI:-/e/project1/transfernetx/$USER/code/envs/snowball-v2/bin/wandb}
    KEYS=${KEYS:-/e/fscratch/reformo/$USER/keys/secrets.env}
    P=jupiter; ACCT=(-A "${SBATCH_ACCOUNT:-laionize}");;
  *) echo "CLUSTER must be horizon or jupiter" >&2; exit 1;;
esac
D=$S/data/kimi0921_pair_v1
EXP=$S/experiments/snowball-kimi0921-pair
CACHE=$EXP/cache-v1
RUN_ID=snowball-kimi0921-pair-$CLUSTER-$REPLICA
OUT=$EXP/$RUN_ID
LOGD=$S/logs/$STAGE; mkdir -p "$LOGD" "$EXP"; LOG=$LOGD/pair_$CLUSTER.log
say() { echo "[$(date -u +%FT%TZ)] [$CLUSTER/$REPLICA] $*" | tee -a "$LOG"; }
die() { say "PAIR_FAILED: $*"; exit 1; }
jobid() { grep -oE 'Submitted batch job [0-9]+' | awk '{print $NF}' | tail -1; }
wait_job() {  # $1 job, $2 log, $3 marker regex
  local st
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
  for _ in 1 2 3 4 5; do st=$(sacct -j "$1" -X -n -o State%20 2>/dev/null | head -1 | awk '{print $1}'); [ -n "$st" ] && break; sleep 20; done
  say "job $1 ended ${st:-?}"
  [ "$st" = COMPLETED ] && grep -qE "$3" "$2"
}
export SNOWBALL_SCRATCH=$S MARIN_ROOT=$MARIN MARIN_PYTHON=$PYM SNOWBALL_STAGE=$STAGE SNOWBALL_TOKENIZER=$BASE
export SNOWBALL_DATASET_ID=$HF_DATA SNOWBALL_DATASET_REVISION=$HF_DATA_REV HF_HUB_OFFLINE=1
[ -f "$BASE/config.json" ] && [ -f "$BASE/training_chat_template.jinja" ] || die "no 09-21 export at $BASE"
say "PAIR_START marin=$(git -C "$MARIN" rev-parse --short HEAD) base=$BASE init=$INIT out=$OUT"

# 1. data: the exact parquet both clusters train on
if [ ! -f "$D/parquet.list" ]; then
  mkdir -p "$D"
  HF_HUB_OFFLINE=0 "$HF" download --repo-type dataset "$HF_DATA" --revision "$HF_DATA_REV" --local-dir "$D" \
    train-00000-of-00001.parquet census.json SHA256SUMS >> "$LOG" 2>&1 || die "HF download of $HF_DATA"
  echo "$PARQUET_SHA  $D/train-00000-of-00001.parquet" | sha256sum -c - >> "$LOG" 2>&1 || die "parquet sha256 mismatch"
  echo "$D/train-00000-of-00001.parquet" > "$D/parquet.list"
fi
say "data ok: $(sha256sum "$D/train-00000-of-00001.parquet" | cut -c1-16)"

# 2. prep (cache) and import (init + sidecar), side by side
jp=; ji=
if [ ! -f "$CACHE/train/.stats.json" ]; then
  jp=$(sbatch "${ACCT[@]}" -o "$LOGD/prep.%j.log" --export=ALL,SNOWBALL_ENV="$(dirname "$(dirname "$PYM")")",SNOWBALL_CACHE="$CACHE",SNOWBALL_PARQUET_LIST="$D/parquet.list" \
       "$MOE/${P}_r2egym_prep.sbatch" 2>&1 | jobid); [ -n "$jp" ] || die "prep submit"; say "prep job $jp"
fi
if [ ! -f "$INIT/snowball_base.json" ]; then
  [ -e "$INIT" ] && die "$INIT exists without snowball_base.json; remove it"
  mkdir -p "$(dirname "$INIT")"
  ji=$(sbatch "${ACCT[@]}" -o "$LOGD/import.%j.log" --export=ALL,SNOWBALL_HF_CHECKPOINT="$BASE",SNOWBALL_INIT="$INIT",SNOWBALL_BASE_CONFIG_FROM_HF=true,SNOWBALL_PENDING_FROM_BIAS=true \
       "$MOE/${P}_snowball_import.sbatch" 2>&1 | jobid); [ -n "$ji" ] || die "import submit"; say "import job $ji"
fi
[ -z "$jp" ] || wait_job "$jp" "$LOGD/prep.$jp.log" "PREP_DONE" || die "prep job $jp"
[ -z "$ji" ] || { wait_job "$ji" "$LOGD/import.$ji.log" "." && [ -f "$INIT/snowball_base.json" ]; } || die "import job $ji"
say "cache: $(head -c 200 "$CACHE/train/.stats.json")"
say "init sidecar: $(tr -d '\n' < "$INIT/snowball_base.json" | head -c 400)"

# 3. run: 30 steps (EPOCHS=1 of the 30-step token-derived epoch), the current recipe's layout and optimizer
if [ ! -f "$LOGD/run_$RUN_ID.done" ]; then
  export SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16 SNOWBALL_NODES=4 SNOWBALL_DEVICES=16
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async OMP_NUM_THREADS=1
  export SNOWBALL_LR=3e-4 SNOWBALL_WARMUP=2 EPOCHS=1 SNOWBALL_SCHEDULE_EPOCHS=1 SNOWBALL_RESUME=0
  export SNOWBALL_INIT=$INIT SNOWBALL_CACHE=$CACHE SNOWBALL_OUTPUT=$OUT SNOWBALL_RUN_ID=$RUN_ID SNOWBALL_WALL=01:00:00
  [ "$CLUSTER" = jupiter ] && export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-laionize}
  out=$(bash "$MOE/launch_${P}_snowball_$([ $P = horizon ] && echo sft || echo r2egym).sh" 2>&1); rc=$?
  echo "$out" >> "$LOG"; echo "$out" | grep -E '^(stage|epochs|steps|layout|lr|init) ' | tee -a "$LOG"
  [ $rc -eq 0 ] || die "launcher rc=$rc"
  jr=$(echo "$out" | jobid); [ -n "$jr" ] || die "no run job id"; say "run job $jr"
  wait_job "$jr" "$S/logs/snowball-$STAGE.$jr.log" "GUARDED_RUN_EXIT rc=0" || die "run job $jr"
  echo "$jr" > "$LOGD/run_$RUN_ID.done"
fi

# 4. W&B: sync the offline run to the pair project and write the per-step loss
WB=$(ls -d "$MARIN"/wandb/offline-run-*-"$RUN_ID" 2>/dev/null | tail -1); [ -n "$WB" ] || die "no offline W&B run for $RUN_ID under $MARIN/wandb"
( set -a; source "$KEYS"; set +a; cd "$(dirname "$WB")" && WANDB_MODE=online "$WANDB" sync --entity "$WANDB_ENTITY_PAIR" --project "$WANDB_PROJECT_PAIR" --id "$RUN_ID" "$(basename "$WB")" ) >> "$LOG" 2>&1 \
  && say "W&B synced: $WANDB_ENTITY_PAIR/$WANDB_PROJECT_PAIR/$RUN_ID" || say "W&B sync FAILED (the TSV below still has the curve)"
"$PYM" - "$WB" "$LOGD/loss_$RUN_ID.tsv" <<'PY' && say "loss TSV: $LOGD/loss_$RUN_ID.tsv"
import glob, json, sys
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore
ds = datastore.DataStore(); ds.open_for_scan(glob.glob(sys.argv[1] + "/*.wandb")[0])
rows = {}
while (d := ds.scan_data()) is not None:
    r = pb.Record(); r.ParseFromString(d)
    if r.WhichOneof("record_type") == "history":
        h = {i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in r.history.item}
        if "train/loss" in h: rows[h["_step"]] = (h["train/loss"], h.get("optim/learning_rate"))
with open(sys.argv[2], "w") as f:
    f.write("step\ttrain_loss\tlr\n")
    for s in sorted(rows): f.write(f"{s}\t{rows[s][0]:.6f}\t{rows[s][1]}\n")
PY
say "PAIR_DONE"
