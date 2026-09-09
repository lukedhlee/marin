#!/bin/bash
# Stage 3 (Nemotron Terminal): weights-only init from the completed Thinking step 630.
#
# Unlike Chat and Thinking, this stage has a MANDATED budget: 1,888 steps, taken
# from STAGES["nemotron_terminal"].fixed_steps rather than derived from the cache.
# Deriving here would be wrong -- 1,888 steps is the reference contract, not an
# epoch. It is read from the stage spec so the launcher and the gate cannot drift.
#
# Set DRY_RUN=1 to print the resolved plan and exit without submitting.
set -euo pipefail

S=/scratch/11694/mkumar73
CACHE=$S/experiments/snowball-chat-vista/nemotron-terminal-cache-v4
INIT=$S/experiments/snowball-chat-vista/thinking-s2/checkpoints/step-630
TOK=$S/hub/hub/models--laion--snowball-67b-a2b-cooldown-step105149/snapshots/b4d2656de8c39b699f7bbcbc878ca96996b319f3
OUT=$S/experiments/snowball-chat-vista/nemotron-terminal-s3
PYBIN=$S/envs/marin-grug-sft-nccl2293/bin/python

[ -d "$INIT" ]  || { echo "FATAL: Thinking step-630 checkpoint missing: $INIT" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "FATAL: Stage 3 cache missing: $CACHE" >&2; exit 2; }
# A superseded cache is marked in place. Refuse it rather than train 64 nodes on it:
# nemotron-terminal-cache-v2 held 3 of 29 shards and this launcher used to point at it.
[ -e "$CACHE/INVALID_DO_NOT_USE.txt" ] && {
  echo "FATAL: $CACHE is marked INVALID:" >&2; sed 's/^/  /' "$CACHE/INVALID_DO_NOT_USE.txt" >&2; exit 2; }
[ -z "$(find "$CACHE" -name '*.tmp.*' -print -quit)" ] || {
  echo "FATAL: $CACHE has unfinalized .tmp shards" >&2; exit 2; }

export PYTHONPATH=$S/marin:$S/marin/lib/levanter/src:$S/marin/lib/haliax/src:$S/marin/lib/marin/src:$S/marin/lib/rigging/src:$S/marin/lib/fray/src:$S/marin/lib/zephyr/src:$S/marin/lib/iris/src
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 HF_HUB_OFFLINE=1

# Budget, init step, and cache identity all come from the stage spec, and the
# cache must prove its provenance before a single node is allocated.
read -r STEPS INIT_STEP <<<"$(cd $S/marin && $S/envs/marin-grug-sft/bin/python - <<PY
from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import (
    STAGES, validate_cache_provenance, validate_native_checkpoint_layout)
s = STAGES["nemotron_terminal"]
assert s.fixed_steps, "nemotron_terminal must declare a fixed step budget"
rec = validate_cache_provenance("$CACHE", "nemotron_terminal", "$TOK")
assert len(rec["shards"]) == s.source_files, "cache provenance shard count != pinned"
validate_native_checkpoint_layout("$INIT", expect_step=s.init_step)
print(s.fixed_steps, s.init_step)
PY
)"
[ -n "${STEPS:-}" ] || { echo "FATAL: could not read fixed_steps from the stage spec" >&2; exit 3; }

# Ask the NCCL library ITSELF, via ncclGetVersion, for the version JAX/XLA will
# load. torch.cuda.nccl.version() reports what PyTorch was compiled against
# (2.28.9 here) and says nothing about the JAX collectives -- reading it would
# falsely implicate the exact release that marin#7344 identifies as broken.
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
    echo "FATAL: NCCL is ${NCCL:-unreadable}; marin#7344 leaks proxy-op slots on aarch64" >&2
    echo "       below 2.29.3-1 and wedges multi-node training. Need >= 2.29.3-1." >&2
    exit 4 ;;
esac

cat <<PLAN
stage           = nemotron_terminal
fixed_steps     = ${STEPS}          (mandated, not derived)
init            = ${INIT}
required step   = ${INIT_STEP}      (verified)
cache           = ${CACHE}          (provenance verified, 29 pinned shards)
output          = ${OUT}            (fresh; optimizer/step reset by weights-only init)
python          = ${PYBIN}
nccl            = ${NCCL}
PLAN

[ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN=1 -- not submitting."; exit 0; }
[ -e "$OUT" ] && { echo "FATAL: $OUT exists; refusing to reuse an output path" >&2; exit 2; }

# The shared guarded sbatch asks for 48h. Stage 2 ran 630 steps in 1:15:11, so
# 1,888 steps is ~3.3-3.8h; a 48h request needs a 48h backfill gap on a queue
# 200+ deep. Override to a wall that still leaves generous headroom.
WALL=${SNOWBALL_WALL:-06:00:00}
sbatch -o $S/logs/snowball-nemotron.%j.log \
  -t "$WALL" \
  -J snowball-nemotron-s3 \
  --export=ALL,MARIN_ROOT=$S/marin,MARIN_PYTHON=$PYBIN,\
SNOWBALL_INIT="$INIT",\
SNOWBALL_CACHE="$CACHE",\
SNOWBALL_TOKENIZER="$TOK",\
SNOWBALL_OUTPUT="$OUT",\
SNOWBALL_RUN_ID=snowball-nemotron-s3,\
SNOWBALL_STEPS="$STEPS",\
SNOWBALL_STAGE=nemotron_terminal,STALL_SECONDS=900 \
  $S/marin/experiments/june_tpu_67b_a2b/moe/vista_snowball_guarded.sbatch
