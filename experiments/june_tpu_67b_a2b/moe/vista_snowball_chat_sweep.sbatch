#!/bin/bash
# Find a working XLA runtime config for Snowball chat on Vista, then train --
# all inside ONE 64-node allocation.
#
# The 67B mesh is (data=8, expert=8) = 64 devices, so this cannot be probed on
# gh-dev; every config guess otherwise costs a full allocation plus queue time.
# Two guesses have already been spent:
#   r7  command_buffer= + latency_hiding (no cap)  -> smoke OK, production hung at step 2
#   r8  command_buffer= only                       -> smoke hung at step 0
# Each candidate below runs a bounded 2-step smoke; a hang is capped by `timeout`
# instead of eating the wall clock. The first candidate that writes a verified
# step-2 checkpoint wins and the chain continues on it.
#SBATCH --partition=gh
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --job-name=snowball-chat-sweep
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082,c636-121,c635-101,c641-061,c611-051,c636-152,c608-042,c634-142,c641-012,c634-052,c621-102,c637-042,c638-101,c638-102,c638-112

set -uo pipefail   # NOT -e: a candidate failing is an expected outcome here.

: "${MARIN_ROOT:?}"; : "${MARIN_PYTHON:?}"
: "${SNOWBALL_INIT:?}"; : "${SNOWBALL_CACHE:?}"; : "${SNOWBALL_TOKENIZER:?}"
: "${SNOWBALL_OUTPUT_ROOT:?Set SNOWBALL_OUTPUT_ROOT for per-candidate smoke dirs.}"
: "${SNOWBALL_PROD_OUTPUT:?}"; : "${SNOWBALL_PROD_RUN_ID:?}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-900}"

module purge; module load gcc/13.2.0 cuda/12.8
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 JAX_PLATFORMS=cuda,cpu
export HF_HUB_OFFLINE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${TACC_CUDA_DIR}"
export TRITON_CC="$(command -v gcc)"; export CC="${TRITON_CC}"; export CXX="$(command -v g++)"
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
S_SCRATCH="${SCRATCH:-$(dirname "${MARIN_ROOT}")}"
export JAX_COMPILATION_CACHE_DIR="${S_SCRATCH}/jax-cache/snowball-chat"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir
export SNOWBALL_HANG_DUMP_SECONDS=600
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG_FILE="${S_SCRATCH}/logs/nccl/${SLURM_JOB_ID}/nccl.%h.%p.log"
mkdir -p "${S_SCRATCH}/logs/nccl/${SLURM_JOB_ID}"
export NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13 NCCL_WIN_ENABLE=0
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src:${PYTHONPATH:-}"
cd "${MARIN_ROOT}"

# Candidate 1 is the hero EP reference runtime verbatim -- the known-working
# combination, not a subset of it. Later candidates subtract one thing at a time.
CAND_NAME=(  "hero-exact" "hero-no-cudaasync" "cap2" "cap1" )
CAND_XLA=(
  "--xla_gpu_enable_command_buffer= --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_experimental_parallel_collective_overlap_limit=4"
  "--xla_gpu_enable_command_buffer= --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_experimental_parallel_collective_overlap_limit=4"
  "--xla_gpu_enable_command_buffer= --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_experimental_parallel_collective_overlap_limit=2"
  "--xla_gpu_enable_command_buffer= --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_experimental_parallel_collective_overlap_limit=1"
)
CAND_ALLOC=( "cuda_async" "" "cuda_async" "cuda_async" )

verify_ckpt() {
  local d="$1"
  [ -f "$d/metadata.json" ] && [ -f "$d/manifest.ocdbt" ] \
    && ls "$d"/manifest.[0-9]* >/dev/null 2>&1 \
    && [ -n "$(find "$d/d" -type f -size +0c -print -quit 2>/dev/null)" ]
}

# Pass = got meaningfully past the step where r7 died, at production geometry.
PROBE_MIN_STEPS="${PROBE_MIN_STEPS:-6}"
steps_reached() { # highest completed step in this candidate's log
  grep -oE "Progress on:train [0-9.]+it/257it" "$1" 2>/dev/null \
    | grep -oE "^Progress on:train [0-9.]+" | awk "{print \$3}" | sort -g | tail -1
}

run_smoke() { # $1=output dir  $2=run id
  "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat preflight \
    --init-checkpoint-path "${SNOWBALL_INIT}" --data-cache-path "${SNOWBALL_CACHE}" \
    --tokenizer-path "${SNOWBALL_TOKENIZER}" --output-path "$1" --run-id "$2" \
    --steps 257 --devices "${SLURM_NTASKS}" || return 1
  timeout --signal=KILL "${SMOKE_TIMEOUT}" \
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=cores \
      "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat train \
        --init-checkpoint-path "${SNOWBALL_INIT}" --data-cache-path "${SNOWBALL_CACHE}" \
        --tokenizer-path "${SNOWBALL_TOKENIZER}" --output-path "$1" --run-id "$2" \
        --steps 257 --devices "${SLURM_NTASKS}" 2>&1 | tee "$1.probe.log"
}

WINNER_XLA=""; WINNER_ALLOC=""; WINNER_NAME=""
for i in "${!CAND_NAME[@]}"; do
  name="${CAND_NAME[$i]}"
  out="${SNOWBALL_OUTPUT_ROOT}/smoke-${name}"
  echo "=== SNOWBALL_SWEEP candidate ${name} ==="
  echo "    XLA_FLAGS=${CAND_XLA[$i]}"
  echo "    XLA_PYTHON_CLIENT_ALLOCATOR=${CAND_ALLOC[$i]:-<default>}"
  rm -rf "$out"
  export XLA_FLAGS="${CAND_XLA[$i]}"
  export JAX_ENABLE_PGLE=false
  if [ -n "${CAND_ALLOC[$i]}" ]; then export XLA_PYTHON_CLIENT_ALLOCATOR="${CAND_ALLOC[$i]}"
  else unset XLA_PYTHON_CLIENT_ALLOCATOR; fi
  start=$SECONDS
  run_smoke "$out" "snowball-sweep-${name}"
  rc=$?
  reached="$(steps_reached "$out.probe.log")"; reached="${reached:-0}"
  echo "SNOWBALL_SWEEP_STEPS ${name} reached=${reached} (need >= ${PROBE_MIN_STEPS})"
  if awk "BEGIN{exit !(${reached} >= ${PROBE_MIN_STEPS})}"; then
    echo "SNOWBALL_SWEEP_PASS ${name} reached ${reached} steps in $((SECONDS-start))s"
    WINNER_XLA="${CAND_XLA[$i]}"; WINNER_ALLOC="${CAND_ALLOC[$i]}"; WINNER_NAME="${name}"
    break
  fi
  echo "SNOWBALL_SWEEP_FAIL ${name} reached=${reached} rc=${rc} after $((SECONDS-start))s (137 = hit the ${SMOKE_TIMEOUT}s cap)"
done

if [ -z "${WINNER_NAME}" ]; then
  echo "SNOWBALL_SWEEP_EXHAUSTED: no candidate produced a step-2 checkpoint." >&2
  exit 3
fi

echo "=== SNOWBALL_SWEEP_WINNER ${WINNER_NAME} (reached ${reached} steps at 257-step geometry) ==="
echo "=== starting ${SNOWBALL_PROD_RUN_ID} ==="
export XLA_FLAGS="${WINNER_XLA}"
if [ -n "${WINNER_ALLOC}" ]; then export XLA_PYTHON_CLIENT_ALLOCATOR="${WINNER_ALLOC}"
else unset XLA_PYTHON_CLIENT_ALLOCATOR; fi
set -e
"${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat preflight \
  --init-checkpoint-path "${SNOWBALL_INIT}" --data-cache-path "${SNOWBALL_CACHE}" \
  --tokenizer-path "${SNOWBALL_TOKENIZER}" --output-path "${SNOWBALL_PROD_OUTPUT}" \
  --run-id "${SNOWBALL_PROD_RUN_ID}" --steps 257 --devices "${SLURM_NTASKS}"
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --cpu-bind=cores \
  "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat train \
    --init-checkpoint-path "${SNOWBALL_INIT}" --data-cache-path "${SNOWBALL_CACHE}" \
    --tokenizer-path "${SNOWBALL_TOKENIZER}" --output-path "${SNOWBALL_PROD_OUTPUT}" \
    --run-id "${SNOWBALL_PROD_RUN_ID}" --steps 257 --devices "${SLURM_NTASKS}"
