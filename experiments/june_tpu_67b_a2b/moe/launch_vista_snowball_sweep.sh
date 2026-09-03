#!/bin/bash
# Submit the one-allocation XLA config sweep + Snowball chat production run.
set -euo pipefail
S=/scratch/11694/mkumar73
HF="${S}/hub/hub/models--laion--snowball-67b-a2b-cooldown-step105149/snapshots/b4d2656de8c39b699f7bbcbc878ca96996b319f3"
sbatch \
  -J snowball-chat-sweep \
  -o "${S}/logs/snowball-chat-sweep.%j.log" \
  --export=ALL,\
MARIN_ROOT="${S}/marin",\
MARIN_PYTHON="${S}/envs/marin-grug-sft/bin/python",\
SNOWBALL_INIT="${S}/experiments/snowball-chat-vista/base-native-bufferfix/checkpoints/step-0",\
SNOWBALL_CACHE="${S}/experiments/snowball-chat-vista/wildchat-delphi-chat-cache",\
SNOWBALL_TOKENIZER="${HF}",\
SNOWBALL_OUTPUT_ROOT="${S}/experiments/snowball-chat-vista/sweep",\
SNOWBALL_PROD_OUTPUT="${S}/experiments/snowball-chat-vista/chat-257-r7",\
SNOWBALL_PROD_RUN_ID=snowball-chat-r7,\
SMOKE_TIMEOUT=900 \
  "${S}/marin/experiments/june_tpu_67b_a2b/moe/vista_snowball_chat_sweep.sbatch"
