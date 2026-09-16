#!/bin/bash
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --job-name=snowball-chat

# Jupiter (JSC) port of the Vista script of the same name. Every delta is Jupiter geometry or
# Jupiter environment; the Python launcher is untouched. Jupiter nodes carry FOUR GH200s, but the
# launcher's contract is one JAX device per Slurm rank, so a node runs four one-GPU ranks
# (--ntasks-per-node=4, --gpu-bind=none): levanter's LevanterSlurmCluster splits CUDA_VISIBLE_DEVICES
# across the local tasks by SLURM_LOCALID, and 16 nodes give the same 64 ranks as Vista.
# Modules: Stages/2026 GCC/14.3.0 CUDA/13 (the CUDA module exports CUDA_HOME). Compute nodes have no
# internet: the env is built on a login node by jupiter_snowball_env.sh. Data, caches, checkpoints
# and logs live under SNOWBALL_SCRATCH (/e/data1/mmlaion/<user>/snowball-sft), never fscratch.

# Run with SNOWBALL_MODE=runtime-probe on one GH200 to validate GPU kernels and
# the real cache read, SNOWBALL_MODE=distributed-probe and `sbatch --nodes=2`
# (8 ranks) for the multi-node fabric gate, or SNOWBALL_MODE=import and `sbatch --nodes=8`
# for the memory-sharded HF-to-native conversion. SNOWBALL_MODE=train-chain runs
# smoke, fresh-process resume, and production without releasing the allocation.
# The default mode launches one Chat stage.

set -euo pipefail

: "${MARIN_ROOT:?Set MARIN_ROOT to the Marin checkout.}"
: "${MARIN_PYTHON:?Set MARIN_PYTHON to the Levanter GPU Python executable.}"

SNOWBALL_MODE="${SNOWBALL_MODE:-train}"

module purge 2>/dev/null; module load Stages/2026 GCC/14.3.0 CUDA/13
# Jupiter compute nodes have no /usr/bin/git, and levanter's tracker imports GitPython at import time,
# which refuses to load without a git executable: load the git module and tell GitPython not to insist.
module load git 2>/dev/null || true
export GIT_PYTHON_REFRESH=quiet WANDB_DISABLE_GIT=true
ulimit -c 0   # an aborted 64-rank JAX job dumps hundreds of GB of cores into the cwd
# Slurm's gres plugin sets CUDA_VISIBLE_DEVICES=0,1,2,3 for every task; the fallback keeps the
# per-rank split well defined if a step is ever launched without it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export JAX_PLATFORMS=cuda,cpu
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:?CUDA/13 module did not set CUDA_HOME}"
export TRITON_CC="$(command -v gcc)"
export CC="${TRITON_CC}"
export CXX="$(command -v g++)"
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
# Vista is ONE GH200 per host, so every expert-axis collective crosses the
# InfiniBand fabric. XLA GPU command buffers capture those NCCL collectives
# into a CUDA graph, which hangs the first training step with no error and the
# GPUs pinned at high SM utilization -- the signature in
# https://github.com/marin-community/marin/issues/5675 and the observed stall in
# jobs 963075 and 963707. Both proven Grug MoE GPU trainers
# (experiments/grug/moe_hero_ep, experiments/grug/moe_hero_fsdp) disable command
# buffers for exactly this reason.
# TODO(https://github.com/marin-community/marin/issues/5675): drop the command
# buffer override once the CUDA graph failure is fixed.
#
# Only the topology-independent parts of the EP64 hero runtime are adopted here.
# That reference is GB200 NVL72 -- 16 workers with FOUR GPUs each -- so its
# `cuda_async` allocator and `parallel_collective_overlap_limit=4` are tuned for
# a 4-GPU intra-host NVLink domain. Vista has one GPU per host behind a single
# IB port, where 4 concurrent overlapping collectives contend for one NIC, and
# the default BFC allocator is what the passing one-GH200 canary (job 963973)
# and the passing 8-host ring MoE probe (job 964442) actually ran on.
export JAX_ENABLE_PGLE=false
# Latency hiding is deliberately NOT enabled here. The EP64 hero reference pairs
# `--xla_gpu_enable_latency_hiding_scheduler=true` with
# `--xla_gpu_experimental_parallel_collective_overlap_limit=4`, and that second
# flag is a CAP on how many collectives overlap at once, not a throughput tweak.
# Enabling the scheduler while dropping its cap -- as the first r7 attempt did --
# lets XLA overlap an unbounded number of collectives, which on Vista all contend
# for one IB port per host. Job 964523 cleared the 2-step smoke and the resume
# gate but hung at step 2 of the 257-step production schedule, whose different
# graph admits a different (deadlocking) overlap schedule.
#
# Correctness first: ship only the command buffer override, which is the flag
# with direct evidence behind it. Set SNOWBALL_XLA_EXTRA to A/B the scheduler
# without editing this file, e.g.
#   SNOWBALL_XLA_EXTRA="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_experimental_parallel_collective_overlap_limit=2"
export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${SNOWBALL_XLA_EXTRA:-} ${XLA_FLAGS:-}"

# Persist compiled executables and XLA autotune results so a restart inside the
# allocation does not recompile the 64-device step, and so compile time is
# separable from execution time when diagnosing a stall.
SNOWBALL_SCRATCH="${SNOWBALL_SCRATCH:?set SNOWBALL_SCRATCH to the data root under /e/data1/mmlaion, never fscratch}"
export SNOWBALL_JAX_CACHE="${SNOWBALL_JAX_CACHE:-${SNOWBALL_SCRATCH}/jax-cache/snowball-chat}"
mkdir -p "${SNOWBALL_JAX_CACHE}"
export JAX_COMPILATION_CACHE_DIR="${SNOWBALL_JAX_CACHE}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir

# A silent hang is the failure mode this run keeps hitting, so make every rank
# self-document instead of relying on an operator noticing dead output.
export SNOWBALL_HANG_DUMP_SECONDS="${SNOWBALL_HANG_DUMP_SECONDS:-900}"

# NCCL_DEBUG=WARN produced zero NCCL lines across a seven-hour hang. Keep INIT
# and NET (COLL is far too noisy at 64 ranks) and send it to per-host files so
# stdout stays readable.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export SNOWBALL_NCCL_LOG_DIR="${SNOWBALL_NCCL_LOG_DIR:-${SNOWBALL_SCRATCH}/logs/nccl/${SLURM_JOB_ID:-manual}}"
mkdir -p "${SNOWBALL_NCCL_LOG_DIR}"
export NCCL_DEBUG_FILE="${SNOWBALL_NCCL_LOG_DIR}/nccl.%h.%p.log"
# NCCL_TIMEOUT is not an NCCL variable (absent from libnccl.so.2), so the
# previous NCCL_TIMEOUT=1800 bought no watchdog at all. Bound the collective
# instead of letting it spin for the whole wall clock.
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
# NCCL 2.28's optional symmetric window registration segfaults on all ranks
# during the first cross-host MoE collective on Vista GH200s (job 964441, 8/8
# ranks SIGSEGV right after "Symmetric VA size=96GB"). The ring backend uses
# ordinary NCCL collectives and does not need the device-initiated window API.
export NCCL_WIN_ENABLE=0
# Jupiter: bootstrap over the IB interface (the setting every working multi-node job here carries).
export NCCL_SOCKET_IFNAME=ib0
# Vista's fabric is NVIDIA/Mellanox InfiniBand -- NCCL selects NET/IB/0/GDRDMA.
# The previous FI_PROVIDER=efa / FI_EFA_* settings were AWS Elastic Fabric
# Adapter knobs with no EFA device on this cluster.
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src:${PYTHONPATH:-}"

test -x "${MARIN_PYTHON}"
cd "${MARIN_ROOT}"

run_train() {
  local init_checkpoint_path="$1"
  local output_path="$2"
  local run_id="$3"
  local steps="$4"
  local expected_output_checkpoint_step="${5:-}"
  local -a preflight_args=(
    --init-checkpoint-path "${init_checkpoint_path}"
    --data-cache-path "${SNOWBALL_CACHE}"
    --tokenizer-path "${SNOWBALL_TOKENIZER}"
    --output-path "${output_path}"
    --run-id "${run_id}"
    --steps "${steps}"
    --devices "${SLURM_NTASKS}"
  )
  if [[ -n "${expected_output_checkpoint_step}" ]]; then
    preflight_args+=(--expected-output-checkpoint-step "${expected_output_checkpoint_step}")
  fi
  "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat preflight "${preflight_args[@]}"
  srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NTASKS}" --ntasks-per-node=4 --gpu-bind=none --cpu-bind=cores \
    "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat train \
      "${preflight_args[@]}"
}

if [[ "${SNOWBALL_MODE}" == "runtime-probe" ]]; then
  : "${SNOWBALL_CACHE:?Set SNOWBALL_CACHE to the packed WildChat cache.}"
  : "${SNOWBALL_TOKENIZER:?Set SNOWBALL_TOKENIZER to the pinned tokenizer directory.}"
  "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat \
    gpu-runtime-probe \
      --data-cache-path "${SNOWBALL_CACHE}" \
      --tokenizer-path "${SNOWBALL_TOKENIZER}"
elif [[ "${SNOWBALL_MODE}" == "distributed-probe" ]]; then
  srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NTASKS}" --ntasks-per-node=4 --gpu-bind=none --cpu-bind=cores \
    "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat \
      distributed-probe --expected-devices "${SLURM_NTASKS}"
elif [[ "${SNOWBALL_MODE}" == "import" ]]; then
  : "${SNOWBALL_HF_CHECKPOINT:?Set SNOWBALL_HF_CHECKPOINT to the pinned local HF snapshot.}"
  : "${SNOWBALL_INIT:?Set SNOWBALL_INIT to the new native step-0 checkpoint path.}"
  srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NTASKS}" --ntasks-per-node=4 --gpu-bind=none --cpu-bind=cores \
    "${MARIN_PYTHON}" -m experiments.june_tpu_67b_a2b.moe.import_snowball_hf \
      --hf_checkpoint "${SNOWBALL_HF_CHECKPOINT}" \
      --output_path "${SNOWBALL_INIT}" \
      --distributed true \
      --expert_axis_size 8 \
      --replica_axis_size 1 \
      --model_axis_size 1
elif [[ "${SNOWBALL_MODE}" == "train" ]]; then
  : "${SNOWBALL_INIT:?Set SNOWBALL_INIT to the native weights-only checkpoint root.}"
  : "${SNOWBALL_CACHE:?Set SNOWBALL_CACHE to the packed WildChat cache.}"
  : "${SNOWBALL_TOKENIZER:?Set SNOWBALL_TOKENIZER to the pinned tokenizer directory.}"
  : "${SNOWBALL_OUTPUT:?Set SNOWBALL_OUTPUT to a unique training output directory.}"
  : "${SNOWBALL_RUN_ID:?Set SNOWBALL_RUN_ID to a unique run name.}"
  : "${SNOWBALL_STEPS:?Set SNOWBALL_STEPS to 2 for smoke or 257 for Chat.}"
  run_train \
    "${SNOWBALL_INIT}" \
    "${SNOWBALL_OUTPUT}" \
    "${SNOWBALL_RUN_ID}" \
    "${SNOWBALL_STEPS}" \
    "${SNOWBALL_EXPECT_OUTPUT_CHECKPOINT_STEP:-}"
elif [[ "${SNOWBALL_MODE}" == "train-chain" ]]; then
  : "${SNOWBALL_INIT:?Set SNOWBALL_INIT to the native weights-only checkpoint root.}"
  : "${SNOWBALL_CACHE:?Set SNOWBALL_CACHE to the packed WildChat cache.}"
  : "${SNOWBALL_TOKENIZER:?Set SNOWBALL_TOKENIZER to the pinned tokenizer directory.}"
  : "${SNOWBALL_SMOKE_OUTPUT:?Set SNOWBALL_SMOKE_OUTPUT to a fresh smoke output directory.}"
  : "${SNOWBALL_SMOKE_RUN_ID:?Set SNOWBALL_SMOKE_RUN_ID to a unique smoke run name.}"
  : "${SNOWBALL_OUTPUT:?Set SNOWBALL_OUTPUT to a fresh production output directory.}"
  : "${SNOWBALL_RUN_ID:?Set SNOWBALL_RUN_ID to a unique production run name.}"
  if [[ "${SNOWBALL_SMOKE_OUTPUT}" == "${SNOWBALL_OUTPUT}" ]]; then
    echo "FATAL: smoke and production outputs must be different." >&2
    exit 2
  fi
  run_train "${SNOWBALL_INIT}" "${SNOWBALL_SMOKE_OUTPUT}" "${SNOWBALL_SMOKE_RUN_ID}" 2
  run_train "${SNOWBALL_INIT}" "${SNOWBALL_SMOKE_OUTPUT}" "${SNOWBALL_SMOKE_RUN_ID}" 3 2
  run_train "${SNOWBALL_INIT}" "${SNOWBALL_OUTPUT}" "${SNOWBALL_RUN_ID}" 257
else
  echo "FATAL: SNOWBALL_MODE must be runtime-probe, distributed-probe, import, train, or train-chain; got ${SNOWBALL_MODE}." >&2
  exit 2
fi
