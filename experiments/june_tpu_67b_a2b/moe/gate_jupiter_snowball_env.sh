#!/bin/bash
# 2-node (8-rank) booster gate for the Jupiter Snowball launch environment.
#
# Exercises everything the 64-node chain depends on except the model itself:
# the module load, every new export, the JAX compilation-cache and NCCL log
# directory creation, PYTHONPATH, srun fan-out, multi-host JAX init, and a real
# cross-host collective. Run this before the 16-node allocation lands so a typo
# in the launch environment fails here in three minutes instead of there.
#
# MARIN_ROOT=<checkout> MARIN_PYTHON=<env>/bin/python SNOWBALL_SCRATCH=/e/data1/mmlaion/$USER/snowball-sft \
#   bash gate_jupiter_snowball_env.sh
#
# Note: this does NOT reproduce the CUDA-graph hang. Job 964442 passed an
# 8-host ring MoE probe with command buffers still enabled, because a 128-token
# 64-hidden probe is too small to trigger it. The real gate for the fix is the
# two-step smoke stage at production geometry inside the 16-node chain.
set -euo pipefail

S=${SNOWBALL_SCRATCH:?}
MARIN=${MARIN_ROOT:?}
PYBIN=${MARIN_PYTHON:?}
mkdir -p "${S}/logs"

sbatch \
  -J snowball-env-gate \
  -p booster -A reformo \
  -N 2 --ntasks-per-node=4 --gres=gpu:4 \
  -t 00:15:00 \
  -o "${S}/logs/snowball-env-gate.%j.log" \
  --export=ALL,\
MARIN_ROOT="${MARIN}",\
MARIN_PYTHON="${PYBIN}",\
SNOWBALL_SCRATCH="${S}",\
SNOWBALL_MODE=distributed-probe \
  "${MARIN}/experiments/june_tpu_67b_a2b/moe/jupiter_snowball_chat.sbatch"
