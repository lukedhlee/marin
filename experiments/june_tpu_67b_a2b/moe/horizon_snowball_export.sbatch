#!/bin/bash
#SBATCH --partition=debug
#SBATCH --account=CCR24067
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --time=00:45:00
#SBATCH --job-name=snowball-export-d4

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

# Native checkpoint -> HF-BF16 export (the vLLM-servable layout), on four CPU ranks (one per
# booster node, JAX_PLATFORMS=cpu) so the fp32 native params plus the bf16 export never have to fit
# one host. Mirrors jupiter_snowball_import.sbatch in the other direction. The exporter is Mrinal's
# export_second_cooldown_stage2.py, whose SNOWBALL_EXPORT_* overrides exist for exactly
# this: it initialises JAX from Slurm when SLURM_NTASKS > 1 and writes the Marin chat
# template and the Snowball generation config into the export.
#
#   sbatch --export=ALL,MARIN_ROOT=<checkout>,MARIN_PYTHON=<env>/bin/python,\
#     SNOWBALL_EXPORT_CHECKPOINT=<run>/checkpoints/step-75,\
#     SNOWBALL_EXPORT_OUTPUT=<run>/export-step75-hf-bf16,\
#     SNOWBALL_EXPORT_TOKENIZER=<stage-3 HF snapshot dir> \
#     experiments/june_tpu_67b_a2b/moe/jupiter_snowball_export.sbatch

set -euo pipefail

: "${MARIN_ROOT:?Set MARIN_ROOT to the Marin checkout.}"
: "${MARIN_PYTHON:?Set MARIN_PYTHON to the Levanter Python executable.}"
: "${SNOWBALL_EXPORT_CHECKPOINT:?Set SNOWBALL_EXPORT_CHECKPOINT to the native step-N directory.}"
: "${SNOWBALL_EXPORT_OUTPUT:?Set SNOWBALL_EXPORT_OUTPUT to the new HF export directory.}"
: "${SNOWBALL_EXPORT_TOKENIZER:?Set SNOWBALL_EXPORT_TOKENIZER to the pinned local tokenizer directory.}"

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
export SNOWBALL_EXPORT_EXPERT_AXIS="${SNOWBALL_EXPORT_EXPERT_AXIS:-1}"
export SNOWBALL_EXPORT_REPLICA_AXIS="${SNOWBALL_EXPORT_REPLICA_AXIS:-1}"
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src"

test -x "${MARIN_PYTHON}"
test -f "${SNOWBALL_EXPORT_CHECKPOINT}/metadata.json" || { echo "FATAL: no metadata.json under ${SNOWBALL_EXPORT_CHECKPOINT}" >&2; exit 2; }
test -f "${SNOWBALL_EXPORT_TOKENIZER}/tokenizer.json" || { echo "FATAL: no tokenizer.json under ${SNOWBALL_EXPORT_TOKENIZER}" >&2; exit 2; }
[ -e "${SNOWBALL_EXPORT_OUTPUT}" ] && { echo "FATAL: ${SNOWBALL_EXPORT_OUTPUT} exists; refusing to overwrite an export" >&2; exit 2; }
cd "${MARIN_ROOT}"

srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=cores \
  "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.export_second_cooldown_stage2

# The export is servable only if its config still names the Grug recipe the fork's vLLM loads.
"${MARIN_PYTHON}" - "${SNOWBALL_EXPORT_OUTPUT}" <<'PYEOF'
import json, os, sys
out = sys.argv[1]
cfg = json.load(open(os.path.join(out, "config.json")))
shards = [f for f in os.listdir(out) if f.endswith(".safetensors")]
assert cfg.get("model_type") == "grug_moe", cfg.get("model_type")
assert cfg.get("architectures") == ["GrugMoeForCausalLM"], cfg.get("architectures")
assert shards, "no safetensors shards written"
assert os.path.isfile(os.path.join(out, "chat_template.jinja")) or "chat_template" in json.load(open(os.path.join(out, "tokenizer_config.json")))
print(f"EXPORT_CHECK_OK shards={len(shards)} model_type={cfg['model_type']}")
PYEOF
