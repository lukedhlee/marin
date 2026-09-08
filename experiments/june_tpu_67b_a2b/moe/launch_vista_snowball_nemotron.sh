#!/bin/bash
# Stage 3 (Nemotron Terminal): weights-only init from the completed Thinking step 630.
#
# Unlike Chat and Thinking, this stage has a MANDATED budget: 1,888 steps, taken
# from STAGES["nemotron_terminal"].fixed_steps rather than derived from the cache.
# Deriving here would be wrong -- 1,888 steps is the reference contract, not an
# epoch. It is read from the stage spec so the launcher and the gate cannot drift.
set -euo pipefail

S=/scratch/11694/mkumar73
CACHE=$S/experiments/snowball-chat-vista/nemotron-terminal-cache-v2
INIT=$S/experiments/snowball-chat-vista/thinking-s2/checkpoints/step-630
TOK=$S/hub/hub/models--laion--snowball-67b-a2b-cooldown-step105149/snapshots/b4d2656de8c39b699f7bbcbc878ca96996b319f3
OUT=$S/experiments/snowball-chat-vista/nemotron-terminal-s3

[ -d "$INIT" ]  || { echo "FATAL: Thinking step-630 checkpoint missing: $INIT" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "FATAL: Stage 3 cache missing: $CACHE" >&2; exit 2; }

export PYTHONPATH=$S/marin:$S/marin/lib/levanter/src:$S/marin/lib/haliax/src:$S/marin/lib/marin/src:$S/marin/lib/rigging/src:$S/marin/lib/fray/src:$S/marin/lib/zephyr/src:$S/marin/lib/iris/src
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 HF_HUB_OFFLINE=1

# The mandated budget and the expected init step both come from the stage spec.
read -r STEPS INIT_STEP <<<"$(cd $S/marin && $S/envs/marin-grug-sft/bin/python - <<'PY'
from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import STAGES
s = STAGES["nemotron_terminal"]
assert s.fixed_steps, "nemotron_terminal must declare a fixed step budget"
print(s.fixed_steps, s.init_step)
PY
)"
[ -n "${STEPS:-}" ] || { echo "FATAL: could not read fixed_steps from the stage spec" >&2; exit 3; }
echo "stage=nemotron_terminal fixed_steps=${STEPS} required_init_step=${INIT_STEP}"

sbatch -o $S/logs/snowball-nemotron.%j.log \
  -J snowball-nemotron-s3 \
  --export=ALL,MARIN_ROOT=$S/marin,MARIN_PYTHON=$S/envs/marin-grug-sft-nccl2293/bin/python,\
SNOWBALL_INIT="$INIT",\
SNOWBALL_CACHE="$CACHE",\
SNOWBALL_TOKENIZER="$TOK",\
SNOWBALL_OUTPUT="$OUT",\
SNOWBALL_RUN_ID=snowball-nemotron-s3,\
SNOWBALL_STEPS="$STEPS",\
SNOWBALL_STAGE=nemotron_terminal,STALL_SECONDS=900 \
  $S/marin/experiments/june_tpu_67b_a2b/moe/vista_snowball_guarded.sbatch
