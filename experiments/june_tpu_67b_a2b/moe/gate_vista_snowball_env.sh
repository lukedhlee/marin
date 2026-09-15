#!/bin/bash
# 8-node gh-dev gate for the Vista Snowball launch environment.
#
# Exercises everything the 64-node chain depends on except the model itself:
# the module load, every new export, the JAX compilation-cache and NCCL log
# directory creation, PYTHONPATH, srun fan-out, multi-host JAX init, and a real
# cross-host collective. Run this before the 64-node allocation lands so a typo
# in the launch environment fails here in three minutes instead of there.
#
# Note: this does NOT reproduce the CUDA-graph hang. Job 964442 passed an
# 8-host ring MoE probe with command buffers still enabled, because a 128-token
# 64-hidden probe is too small to trigger it. The real gate for the fix is the
# two-step smoke stage at production geometry inside the 64-node chain.
set -euo pipefail

S=${SCRATCH:?}

sbatch \
  -J snowball-env-gate \
  -p gh-dev \
  -N 8 \
  -t 00:15:00 \
  -o "${S}/logs/snowball-env-gate.%j.log" \
  --export=ALL,\
MARIN_ROOT="${S}/marin",\
MARIN_PYTHON="${S}/envs/marin-grug-sft/bin/python",\
SNOWBALL_MODE=distributed-probe \
  "${S}/marin/experiments/june_tpu_67b_a2b/moe/vista_snowball_chat.sbatch"
