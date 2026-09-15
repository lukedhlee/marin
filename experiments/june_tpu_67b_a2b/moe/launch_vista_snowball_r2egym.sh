#!/bin/bash
# R2E-Gym teacher-trace SFT of the PRE-RL Snowball checkpoint on Vista (stage "r2egym").
#
# Order of operations, all under $SCRATCH (see vista_r2egym_prep.sbatch for the first two):
#   1. prep      sbatch vista_r2egym_prep.sbatch      env + packed cache + probes (1 gh-dev node)
#   2. gate      gate_vista_snowball_env.sh            8-node fabric gate (paths inside are Mrinal's;
#                                                      pass MARIN_ROOT/MARIN_PYTHON via --export)
#   3. import    sbatch vista_snowball_import.sbatch   HF export -> native step-0 (4 gg CPU nodes)
#   4. this      EPOCHS=3 launch_vista_snowball_r2egym.sh
#
# The step budget is EPOCHS x the epoch derived from the built cache (~26 packed steps per epoch
# for the 2,576-trace set); the stage's max_steps caps it at five epochs. Set DRY_RUN=1 to print
# the resolved plan and exit.
set -euo pipefail

S=${SCRATCH:?}
EXP=$S/experiments/snowball-r2egym-sft
CACHE=${SNOWBALL_CACHE:-$EXP/cache-v1}
INIT=${SNOWBALL_INIT:-$EXP/init-s3-step1888}
TOK=${SNOWBALL_TOKENIZER:?set to the Stage-3 HF snapshot dir (holds tokenizer.json + chat_template.jinja)}
OUT=${SNOWBALL_OUTPUT:-$EXP/r2egym-glm47-solved-v1-run1}
MARIN=${MARIN_ROOT:-$S/marin}
PYBIN=${MARIN_PYTHON:-$S/envs/marin-grug-sft/bin/python}
EPOCHS=${EPOCHS:-3}
WALL=${SNOWBALL_WALL:-03:00:00}
RUN_ID=${SNOWBALL_RUN_ID:-snowball-r2egym-sft-run1}

[ -d "$INIT" ]  || { echo "FATAL: imported native init missing: $INIT" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "FATAL: r2egym cache missing: $CACHE" >&2; exit 2; }
[ -e "$CACHE/INVALID_DO_NOT_USE.txt" ] && { echo "FATAL: $CACHE is marked INVALID" >&2; exit 2; }
[ -z "$(find "$CACHE" -name '*.tmp.*' -print -quit)" ] || { echo "FATAL: $CACHE has unfinalized .tmp shards" >&2; exit 2; }

export PYTHONPATH=$MARIN:$MARIN/lib/levanter/src:$MARIN/lib/haliax/src:$MARIN/lib/marin/src:$MARIN/lib/rigging/src:$MARIN/lib/fray/src:$MARIN/lib/zephyr/src:$MARIN/lib/iris/src
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 HF_HUB_OFFLINE=1

# Budget from the cache the stage actually built; provenance and init step from the stage spec.
read -r STEPS EPOCH_STEPS INIT_STEP <<<"$(cd $MARIN && JAX_PLATFORMS=cpu $PYBIN - <<PY
from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import (
    STAGES, derive_epoch_steps, read_chat_cache_tokens, validate_cache_provenance,
    validate_native_checkpoint_layout)
s = STAGES["r2egym"]
rec = validate_cache_provenance("$CACHE", "r2egym", "$TOK")
assert len(rec["shards"]) == s.source_files, "cache provenance shard count != pinned"
validate_native_checkpoint_layout("$INIT", expect_step=s.init_step)
epoch = derive_epoch_steps(read_chat_cache_tokens("$CACHE"))
steps = $EPOCHS * epoch
assert s.max_steps is None or steps <= s.max_steps, f"{steps} steps exceeds the stage ceiling {s.max_steps}"
print(steps, epoch, s.init_step)
PY
)"
[ -n "${STEPS:-}" ] || { echo "FATAL: could not derive the step budget" >&2; exit 3; }

# Ask the NCCL library itself which release JAX will load (marin #7344: < 2.29.3 wedges on aarch64).
NCCL=$($PYBIN - <<'PY' 2>/dev/null
import ctypes, glob, os, sys
sos = sorted(glob.glob(os.path.join(sys.prefix, "**", "libnccl.so.2"), recursive=True))
if not sos:
    print("UNKNOWN"); raise SystemExit
lib = ctypes.CDLL(sos[0]); v = ctypes.c_int()
lib.ncclGetVersion(ctypes.byref(v))
major, rest = divmod(v.value, 10000); minor, patch = divmod(rest, 100)
print(f"{major}.{minor}.{patch}")
PY
)
case "$NCCL" in
  2.2[0-8].*|UNKNOWN|"")
    echo "FATAL: NCCL is ${NCCL:-unreadable}; need >= 2.29.3-1 (marin #7344)" >&2; exit 4 ;;
esac

cat <<PLAN
stage           = r2egym
epochs          = ${EPOCHS}  (${EPOCH_STEPS} packed steps per epoch)
steps           = ${STEPS}
init            = ${INIT}   (required step ${INIT_STEP}, verified)
cache           = ${CACHE}  (provenance verified)
output          = ${OUT}    (fresh; optimizer/step reset by weights-only init)
python          = ${PYBIN}
nccl            = ${NCCL}
wall            = ${WALL}
PLAN

[ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN=1 -- not submitting."; exit 0; }
[ -e "$OUT" ] && { echo "FATAL: $OUT exists; refusing to reuse an output path" >&2; exit 2; }
mkdir -p $S/logs

sbatch -o $S/logs/snowball-r2egym.%j.log \
  -t "$WALL" \
  -J "$RUN_ID" \
  --export=ALL,MARIN_ROOT=$MARIN,MARIN_PYTHON=$PYBIN,\
SNOWBALL_INIT="$INIT",\
SNOWBALL_CACHE="$CACHE",\
SNOWBALL_TOKENIZER="$TOK",\
SNOWBALL_OUTPUT="$OUT",\
SNOWBALL_RUN_ID="$RUN_ID",\
SNOWBALL_STEPS="$STEPS",\
SNOWBALL_STAGE=r2egym,STALL_SECONDS=900 \
  $MARIN/experiments/june_tpu_67b_a2b/moe/vista_snowball_guarded.sbatch
