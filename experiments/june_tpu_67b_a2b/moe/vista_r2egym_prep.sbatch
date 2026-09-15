#!/bin/bash
# One-node preparation for the R2E-Gym SFT of the pre-RL Snowball on Vista, in one gh-dev
# allocation (2h cap): build the Levanter GPU env from this checkout's lock, override NCCL to a
# release with the aarch64 proxy-slot fix (marin #7344 -> >= 2.29.3), build the packed chat cache
# for the r2egym stage, record its provenance, and run the GPU kernel + runtime probes against it.
#
# Every step is idempotent: an existing env is reused (sync is a no-op), an existing cache is
# validated instead of rebuilt. Set PREP_STEPS to a subset ("env", "cache", "probe") to rerun one.
#
#   sbatch --export=ALL,MARIN_ROOT=$SCRATCH/marin,SNOWBALL_ENV=$SCRATCH/envs/marin-grug-sft,\
#     SNOWBALL_TOKENIZER=<stage-3 HF snapshot dir>,SNOWBALL_CACHE=$SCRATCH/experiments/snowball-r2egym-sft/cache-v1,\
#     SNOWBALL_PARQUET_LIST=$SCRATCH/data/r2egym_glm47_solved_v1/parquet.list \
#     experiments/june_tpu_67b_a2b/moe/vista_r2egym_prep.sbatch
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --job-name=snowball-r2egym-prep

set -uo pipefail
: "${MARIN_ROOT:?}"; : "${SNOWBALL_ENV:?}"; : "${SNOWBALL_TOKENIZER:?}"
: "${SNOWBALL_CACHE:?}"; : "${SNOWBALL_PARQUET_LIST:?}"
PREP_STEPS="${PREP_STEPS:-env cache probe}"
NCCL_OVERRIDE="${NCCL_OVERRIDE:-2.30.7}"   # the release Mrinal's completed stages ran on
S_SCRATCH="${SCRATCH:-$(dirname "${MARIN_ROOT}")}"

module purge; module load gcc/13.2.0 cuda/12.8
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 JAX_PLATFORMS=cuda,cpu
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export CUDA_HOME="${TACC_CUDA_DIR}"
export TRITON_CC="$(command -v gcc)"; export CC="${TRITON_CC}"; export CXX="$(command -v g++)"
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
export JAX_ENABLE_PGLE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${SNOWBALL_XLA_EXTRA:-}"
export JAX_COMPILATION_CACHE_DIR="${S_SCRATCH}/jax-cache/snowball-r2egym"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${S_SCRATCH}/logs"
export UV_CACHE_DIR="${S_SCRATCH}/.cache/uv" UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="${SNOWBALL_ENV}"
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src"
PY="${SNOWBALL_ENV}/bin/python"
cd "${MARIN_ROOT}"
echo "PREP_START host=$(hostname) steps='${PREP_STEPS}' env=${SNOWBALL_ENV} nccl_override=${NCCL_OVERRIDE}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

step_env() {
  [ -x "${PY}" ] || uv venv --python 3.12 "${SNOWBALL_ENV}" || return 1
  # The workspace lock, unchanged: the same JAX/FA4/cuTe pins Mrinal's completed stages ran on.
  uv sync --all-packages --extra=gpu --frozen || return 1
  # marin #7344: NCCL 2.28.9 (the lock's pin) leaks proxy-op slots on aarch64 and wedges silently.
  uv pip install --python "${PY}" --no-deps "nvidia-nccl-cu13==${NCCL_OVERRIDE}" || return 1
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
      --stage r2egym --parquet-list "${SNOWBALL_PARQUET_LIST}" --expect-files 1 \
      --cache-path "${SNOWBALL_CACHE}" --tokenizer-path "${SNOWBALL_TOKENIZER}" || return 1
  fi
  "${PY}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat write-provenance \
    --cache-path "${SNOWBALL_CACHE}" --stage r2egym \
    --dataset-id "DCAgent/g1_clean_hybrid_scaffold_plus_r2eg_gfi_38k_glm47_traces" \
    --dataset-revision "4243a8f5cd39799803a6a0d52457fa0833068566" \
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
