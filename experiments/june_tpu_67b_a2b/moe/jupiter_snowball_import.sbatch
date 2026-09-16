#!/bin/bash
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:45:00
#SBATCH --job-name=snowball-base-import-d4

# Jupiter (JSC) port of the Vista script of the same name. Every delta is Jupiter geometry or
# Jupiter environment; the Python launcher is untouched. Jupiter nodes carry FOUR GH200s, but the
# launcher's contract is one JAX device per Slurm rank, so a node runs four one-GPU ranks
# (--ntasks-per-node=4, --gpu-bind=none): levanter's LevanterSlurmCluster splits CUDA_VISIBLE_DEVICES
# across the local tasks by SLURM_LOCALID, and 16 nodes give the same 64 ranks as Vista.
# Modules: Stages/2026 GCC/14.3.0 CUDA/13 (the CUDA module exports CUDA_HOME). Compute nodes have no
# internet: the env is built on a login node by jupiter_snowball_env.sh. Data, caches, checkpoints
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

module purge 2>/dev/null
module load Stages/2026 GCC/14.3.0
# Jupiter compute nodes have no /usr/bin/git, and levanter's tracker imports GitPython at import time,
# which refuses to load without a git executable: load the git module and tell GitPython not to insist.
module load git 2>/dev/null || true
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
