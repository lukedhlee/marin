#!/bin/bash
# Submit the 64-node Snowball Base-to-Chat SFT chain on TACC Vista.
#
# Carries the fix from commit e35f99160: XLA GPU command buffers are disabled,
# so the first training step no longer hangs capturing cross-host NCCL
# collectives into a CUDA graph (marin issue 5675). Jobs 963075 and 963707 both
# stalled on that, at the same line, producing no loss and no checkpoint.
#
# Fresh output directories -- chat-smoke-r6 and chat-257-r4 from the previous
# attempts are never written to again.
set -euo pipefail

S=/scratch/11694/mkumar73
HF_SNAPSHOT="${S}/hub/hub/models--laion--snowball-67b-a2b-cooldown-step105149/snapshots/b4d2656de8c39b699f7bbcbc878ca96996b319f3"

sbatch \
  -J snowball-chat-cbfix-r8 \
  -o "${S}/logs/snowball-chat-cbfix-r8.%j.log" \
  --export=ALL,\
MARIN_ROOT="${S}/marin",\
MARIN_PYTHON="${S}/envs/marin-grug-sft/bin/python",\
SNOWBALL_MODE=train-chain,\
SNOWBALL_INIT="${S}/experiments/snowball-chat-vista/base-native-bufferfix/checkpoints/step-0",\
SNOWBALL_CACHE="${S}/experiments/snowball-chat-vista/wildchat-delphi-chat-cache",\
SNOWBALL_TOKENIZER="${HF_SNAPSHOT}",\
SNOWBALL_SMOKE_OUTPUT="${S}/experiments/snowball-chat-vista/chat-smoke-r8",\
SNOWBALL_SMOKE_RUN_ID=snowball-chat-smoke-r8,\
SNOWBALL_OUTPUT="${S}/experiments/snowball-chat-vista/chat-257-r6",\
SNOWBALL_RUN_ID=snowball-chat-r6 \
  "${S}/marin/experiments/june_tpu_67b_a2b/moe/vista_snowball_chat.sbatch"
