#!/bin/bash
# Jupiter (JSC) port of the Vista script of the same name. Every delta is Jupiter geometry or
# Jupiter environment; the Python launcher is untouched. Jupiter nodes carry FOUR GH200s, but the
# launcher's contract is one JAX device per Slurm rank, so a node runs four one-GPU ranks
# (--ntasks-per-node=4, --gpu-bind=none): levanter's LevanterSlurmCluster splits CUDA_VISIBLE_DEVICES
# across the local tasks by SLURM_LOCALID, and 16 nodes give the same 64 ranks as Vista.
# Modules: Stages/2026 GCC/14.3.0 CUDA/13 (the CUDA module exports CUDA_HOME). Compute nodes have no
# internet: the env is built on a login node by horizon_snowball_env.sbatch. Data, caches, checkpoints
# and logs live under SNOWBALL_SCRATCH (/e/data1/mmlaion/<user>/snowball-sft), never fscratch.
#
# One-node preparation for the R2E-Gym SFT of the pre-RL Snowball on Jupiter: check the env that
# horizon_snowball_env.sbatch built on the login node (NCCL >= 2.29.3, marin #7344; JAX sees the GPU),
# build the packed chat cache for the r2egym stage, record its provenance, and run the GPU kernel +
# runtime probes against it.
#
# Every step is idempotent: an existing cache is validated instead of rebuilt. Set PREP_STEPS to a
# subset ("envcheck", "cache", "probe") to rerun one.
#
#   S=/e/data1/mmlaion/$USER/snowball-sft
#   sbatch --export=ALL,MARIN_ROOT=<marin checkout>,SNOWBALL_ENV=<env dir>,SNOWBALL_SCRATCH=$S,\
#     SNOWBALL_TOKENIZER=<stage-3 HF snapshot dir>,SNOWBALL_CACHE=$S/experiments/snowball-r2egym-sft/cache-v1,\
#     SNOWBALL_PARQUET_LIST=$S/data/r2egym_glm47_solved_v1/parquet.list \
#     experiments/june_tpu_67b_a2b/moe/jupiter_r2egym_prep.sbatch
#SBATCH --partition=debug
#SBATCH --account=CCR24067
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --time=01:00:00
#SBATCH --job-name=snowball-r2egym-prep

# Horizon (TACC, 4x GB200 per node, aarch64 sm_100) port of the jupiter_* script of the same name. Every delta is
# Horizon geometry or environment; the Python launcher is untouched: partition debug / account CCR24067, 144 CPUs
# per node (36 per rank at four ranks), system GCC 14 instead of modules (TACC's default nvidia/26.9 sets CC=nvc,
# which breaks Triton/XLA host compiles), CUDA_HOME from the NVHPC 26.9 CUDA 13.3 tree, NCCL bootstrap on the IPoIB
# interfaces (ib*), git present on compute nodes. The env is built by horizon_snowball_env.sbatch on a compute node.
# Data, caches, checkpoints and logs live under SNOWBALL_SCRATCH (/scratch/11584/<user>/snowball-sft).

set -uo pipefail
: "${MARIN_ROOT:?}"; : "${SNOWBALL_ENV:?}"; : "${SNOWBALL_TOKENIZER:?}"
: "${SNOWBALL_CACHE:?}"; : "${SNOWBALL_PARQUET_LIST:?}"
PREP_STEPS="${PREP_STEPS:-envcheck cache probe}"
NCCL_OVERRIDE="${NCCL_OVERRIDE:-2.30.7}"   # the release Mrinal's completed stages ran on
S_SCRATCH="${SNOWBALL_SCRATCH:?set SNOWBALL_SCRATCH to the data root under /e/data1/mmlaion, e.g. /scratch/11584/$USER/snowball-sft}"

export CC=gcc CXX=g++   # Horizon: system GCC 14 (the login env's nvidia/26.9 exports CC=nvc)
export CUDA_HOME="${CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/26.9/cuda/13.3}"
# Jupiter compute nodes have no /usr/bin/git, and levanter's tracker imports GitPython at import time,
# which refuses to load without a git executable: load the git module and tell GitPython not to insist.
export GIT_PYTHON_REFRESH=quiet WANDB_DISABLE_GIT=true
ulimit -c 0   # an aborted 64-rank JAX job dumps hundreds of GB of cores into the cwd
# The probes run in the batch step itself (no srun): pin them to one GPU as on a Vista node.
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 JAX_PLATFORMS=cuda,cpu
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export TRITON_CC="$(command -v gcc)"; export CC="${TRITON_CC}"; export CXX="$(command -v g++)"
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
export JAX_ENABLE_PGLE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${SNOWBALL_XLA_EXTRA:-}"
# The stage names the STAGES entry the cache is built for; the dataset id / revision are recorded in
# the cache's provenance and must match what the stage pins (r2egym defaults, any other stage sets both).
SFT_STAGE="${SNOWBALL_STAGE:-r2egym}"
DATASET_ID="${SNOWBALL_DATASET_ID:-DCAgent/g1_clean_hybrid_scaffold_plus_r2eg_gfi_38k_glm47_traces}"
DATASET_REVISION="${SNOWBALL_DATASET_REVISION:-4243a8f5cd39799803a6a0d52457fa0833068566}"
export JAX_COMPILATION_CACHE_DIR="${S_SCRATCH}/jax-cache/snowball-${SFT_STAGE}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${S_SCRATCH}/logs"
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src"
PY="${SNOWBALL_ENV}/bin/python"
cd "${MARIN_ROOT}"
echo "PREP_START host=$(hostname) steps='${PREP_STEPS}' env=${SNOWBALL_ENV} nccl_override=${NCCL_OVERRIDE}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

step_envcheck() {
  # The env itself is built by horizon_snowball_env.sbatch on a login node (uv needs the internet).
  [ -x "${PY}" ] || { echo "no python at ${PY}; run horizon_snowball_env.sbatch on a login node first" >&2; return 1; }
  "${PY}" - <<'PYEOF' || return 1
import ctypes, glob, os, sys
sos = sorted(glob.glob(os.path.join(sys.prefix, "**", "libnccl.so.2"), recursive=True))
assert sos, "no libnccl.so.2 in the env"
lib = ctypes.CDLL(sos[0]); v = ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(v))
major, rest = divmod(v.value, 10000); minor, patch = divmod(rest, 100)
print(f"nccl_runtime={major}.{minor}.{patch} ({sos[0]})")
assert (major, minor, patch) >= (2, 29, 3), "NCCL below 2.29.3 -- marin #7344"
import jax; print("jax", jax.__version__, "devices", jax.devices())
PYEOF
}

step_cache() {
  if [ -f "${SNOWBALL_CACHE}/train/.stats.json" ]; then
    echo "cache exists at ${SNOWBALL_CACHE}; validating instead of rebuilding"
  else
    "${PY}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat prepare-data \
      --stage "${SFT_STAGE}" --parquet-list "${SNOWBALL_PARQUET_LIST}" --expect-files 1 \
      --cache-path "${SNOWBALL_CACHE}" --tokenizer-path "${SNOWBALL_TOKENIZER}" || return 1
  fi
  "${PY}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat write-provenance \
    --cache-path "${SNOWBALL_CACHE}" --stage "${SFT_STAGE}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-revision "${DATASET_REVISION}" \
    --parquet-list "${SNOWBALL_PARQUET_LIST}" --expect-files 1 \
    --tokenizer-path "${SNOWBALL_TOKENIZER}" || return 1
  "${PY}" - "${SNOWBALL_CACHE}" <<'PYEOF' || return 1
import sys
from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import derive_epoch_steps, read_chat_cache_tokens, validate_chat_cache_layout
tokens, examples, shards = validate_chat_cache_layout(sys.argv[1], expect_tokens=None, expect_examples=None)
print(f"cache_tokens={tokens} cache_examples={examples} cache_shards={shards} epoch_steps={derive_epoch_steps(read_chat_cache_tokens(sys.argv[1]))}")
PYEOF
}

step_probe() {
  "${PY}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat gpu-kernel-probe || return 1
  "${PY}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat gpu-runtime-probe \
    --data-cache-path "${SNOWBALL_CACHE}" --tokenizer-path "${SNOWBALL_TOKENIZER}" || return 1
}

for step in ${PREP_STEPS}; do
  echo "=== PREP_STEP ${step} start $(date -u +%FT%TZ)"
  if "step_${step}"; then echo "=== PREP_STEP ${step} OK"; else echo "=== PREP_STEP ${step} FAILED"; exit 2; fi
done
echo "PREP_DONE"
