#!/bin/bash
#SBATCH --partition=debug
#SBATCH --account=CCR24067
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --time=00:45:00
#SBATCH --job-name=snowball-base-import-d4

# Horizon (TACC, 4x GB200 per node, aarch64 sm_100) port of the jupiter_* script of the same name. Every delta is
# Horizon geometry or environment; the Python launcher is untouched: partition debug / account CCR24067, 144 CPUs
# per node (36 per rank at four ranks), system GCC 14 instead of modules (TACC's default nvidia/26.9 sets CC=nvc,
# which breaks Triton/XLA host compiles), CUDA_HOME from the NVHPC 26.9 CUDA 13.3 tree, NCCL bootstrap on the IB
# interfaces (ib*), git present on compute nodes. The env is built by horizon_snowball_env.sbatch on a compute node.
# Data, caches, checkpoints and logs live under SNOWBALL_SCRATCH (/scratch/11584/<user>/snowball-sft).

# Jupiter (JSC) port of the Vista script of the same name. Every delta is Jupiter geometry or
# Jupiter environment; the Python launcher is untouched. Jupiter nodes carry FOUR GH200s, but the
# launcher's contract is one JAX device per Slurm rank, so a node runs four one-GPU ranks
# (--ntasks-per-node=4, --gpu-bind=none): levanter's LevanterSlurmCluster splits CUDA_VISIBLE_DEVICES
# across the local tasks by SLURM_LOCALID, and 16 nodes give the same 64 ranks as Vista.
# Modules: Stages/2026 GCC/14.3.0 CUDA/13 (the CUDA module exports CUDA_HOME). Compute nodes have no
# internet: the env is built on a login node by horizon_snowball_env.sbatch. Data, caches, checkpoints
# and logs live under SNOWBALL_SCRATCH (/e/data1/mmlaion/<user>/snowball-sft), never fscratch.
#
# Four CPU ranks (JAX_PLATFORMS=cpu, one per booster node; Jupiter has no CPU-only partition, so the
# GH200s idle for the few minutes this takes) give the importer a data axis of four. Each rank holds
# roughly one quarter of the 134 GB source plus one quarter of the converted layout.

set -euo pipefail

: "${MARIN_ROOT:?Set MARIN_ROOT to the Marin checkout.}"
: "${MARIN_PYTHON:?Set MARIN_PYTHON to the Levanter GPU Python executable.}"
: "${SNOWBALL_HF_CHECKPOINT:?Set SNOWBALL_HF_CHECKPOINT to the pinned local HF snapshot.}"
: "${SNOWBALL_INIT:?Set SNOWBALL_INIT to the new native step-0 checkpoint path.}"

export CC=gcc CXX=g++   # Horizon: system GCC 14 (the login env's nvidia/26.9 exports CC=nvc)
export CUDA_HOME="${CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/26.9/cuda/13.3}"
# Jupiter compute nodes have no /usr/bin/git, and levanter's tracker imports GitPython at import time,
# which refuses to load without a git executable: load the git module and tell GitPython not to insist.
export GIT_PYTHON_REFRESH=quiet WANDB_DISABLE_GIT=true
ulimit -c 0
# The wheels want a newer libstdc++ than RHEL 9 ships; preload the GCC module's copy as on the GPU path.
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src:${PYTHONPATH:-}"

test -x "${MARIN_PYTHON}"
cd "${MARIN_ROOT}"

srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=cores \
  "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.import_snowball_hf \
    --hf_checkpoint "${SNOWBALL_HF_CHECKPOINT}" \
    --output_path "${SNOWBALL_INIT}" \
    --distributed true \
    --expert_axis_size 1 \
    --replica_axis_size 1 \
    --model_axis_size 1 \
    --validate_only "${SNOWBALL_VALIDATE_ONLY:-false}"
