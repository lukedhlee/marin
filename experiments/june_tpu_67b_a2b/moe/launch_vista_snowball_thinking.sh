#!/bin/bash
# Stage 2 (Thinking): weights-only init from the completed Chat step-257.
#
# Chains natively -- trainer.initialize_from does a weights-only load and keeps a
# fresh optimizer/step, so NO HF export is involved. HF export is only needed for
# eval and serving, which is a separate branch.
#
# The epoch length is DERIVED from the completed cache at launch time. It is never
# hardcoded, and in particular is not assumed to be the historical 630.
set -euo pipefail

S=/scratch/11694/mkumar73
REV=bae881d7227146ef6b93fe830a1f613e96ea1338
CACHE=$S/experiments/snowball-chat-vista/nemotron-thinking-cache
CHAT=$S/experiments/snowball-chat-vista/chat-257-nccl2307b/checkpoints/step-257
TOK=$S/hub/hub/models--laion--snowball-67b-a2b-cooldown-step105149/snapshots/b4d2656de8c39b699f7bbcbc878ca96996b319f3

[ -d "$CHAT" ]  || { echo "FATAL: Chat step-257 checkpoint missing: $CHAT" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "FATAL: Thinking cache missing: $CACHE  (run build_thinking_cache first)" >&2; exit 2; }

export PYTHONPATH=$S/marin:$S/marin/lib/levanter/src:$S/marin/lib/haliax/src:$S/marin/lib/marin/src:$S/marin/lib/rigging/src:$S/marin/lib/fray/src:$S/marin/lib/zephyr/src:$S/marin/lib/iris/src
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 HF_HUB_OFFLINE=1

# Derive the epoch from the cache that was actually built.
STEPS=$(cd $S/marin && $S/envs/marin-grug-sft/bin/python - <<PY
from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import read_chat_cache_tokens, expected_chat_steps
t = read_chat_cache_tokens("$CACHE")
print(expected_chat_steps(t))
PY
)
[ -n "$STEPS" ] || { echo "FATAL: could not derive epoch length from $CACHE" >&2; exit 3; }
echo "derived epoch length from cache: ${STEPS} steps"

sbatch -o $S/logs/snowball-thinking.%j.log \
  -J snowball-thinking \
  --export=ALL,MARIN_ROOT=$S/marin,MARIN_PYTHON=$S/envs/marin-grug-sft-nccl2293/bin/python,\
SNOWBALL_INIT="$CHAT",\
SNOWBALL_CACHE="$CACHE",\
SNOWBALL_TOKENIZER="$TOK",\
SNOWBALL_OUTPUT=$S/experiments/snowball-chat-vista/thinking-s2,\
SNOWBALL_RUN_ID=snowball-thinking-s2,\
SNOWBALL_STEPS="$STEPS",\
SNOWBALL_STAGE=thinking,STALL_SECONDS=900 \
  $S/marin/experiments/june_tpu_67b_a2b/moe/vista_snowball_guarded.sbatch
