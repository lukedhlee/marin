# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Per-cluster settings for the relay SFT jobs, sourced by every script in this directory. RELAY_CLUSTER picks the
# cluster; RELAY_ROOT (data, caches, checkpoints, logs) and RELAY_PYTHON (the env built by build_env.sh) are
# required everywhere. Jupiter (JSC) nodes carry 4 GH200 and 288 cores, Horizon (TACC) nodes 4 GB200 and 144 cores;
# a training job runs one JAX device per Slurm rank, four ranks per node.

: "${RELAY_CLUSTER:?set RELAY_CLUSTER to jupiter or horizon}"
: "${RELAY_ROOT:?set RELAY_ROOT to the data root (Jupiter: under /e/data1, never fscratch)}"
RELAY_SLURM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MARIN_ROOT=${MARIN_ROOT:-$(cd "$RELAY_SLURM_DIR/../../../.." && pwd)}
RELAY_PYTHON=${RELAY_PYTHON:-$RELAY_ROOT/envs/marin-relay-sft/bin/python}

case "$RELAY_CLUSTER" in
  jupiter)
    RELAY_PARTITION=${RELAY_PARTITION:-booster}
    RELAY_ACCOUNT=${RELAY_ACCOUNT:?set RELAY_ACCOUNT to your Jupiter compute project}
    RELAY_NODE_CPUS=288
    RELAY_NCCL_IFNAME=ib0
    ;;
  horizon)
    RELAY_PARTITION=${RELAY_PARTITION:-debug}
    RELAY_ACCOUNT=${RELAY_ACCOUNT:?set RELAY_ACCOUNT to your TACC allocation}
    RELAY_NODE_CPUS=144
    RELAY_NCCL_IFNAME=ib   # a prefix: ibs2, ibP2p1s0, ...
    ;;
  *) echo "RELAY_CLUSTER must be jupiter or horizon, got $RELAY_CLUSTER" >&2; exit 1 ;;
esac

# The compute-node environment shared by every job: compiler, CUDA, libstdc++, PYTHONPATH, offline HF and W&B.
relay_compute_env() {
  case "$RELAY_CLUSTER" in
    jupiter)
      # non-interactive shells do not define `module`
      type module >/dev/null 2>&1 || source "${LMOD_PKG:-/e/software/default/lmod/8.7.64}/init/bash"
      module purge 2>/dev/null
      module load Stages/2026 GCC/14.3.0 CUDA/13
      # compute nodes have no /usr/bin/git, and the W&B tracker imports GitPython
      module load git 2>/dev/null || true
      ;;
    horizon)
      # the login environment's NVHPC sets CC=nvc, which breaks Triton and XLA host compiles
      export CUDA_HOME=${CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/26.9/cuda/13.3}
      ;;
  esac
  export CC=gcc CXX=g++ TRITON_CC=gcc
  export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
  export GIT_PYTHON_REFRESH=quiet WANDB_DISABLE_GIT=true WANDB_MODE=offline
  export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1
  export PYTHONPATH="$MARIN_ROOT:$MARIN_ROOT/lib/levanter/src:$MARIN_ROOT/lib/haliax/src:$MARIN_ROOT/lib/marin/src:$MARIN_ROOT/lib/rigging/src:$MARIN_ROOT/lib/fray/src:$MARIN_ROOT/lib/zephyr/src:$MARIN_ROOT/lib/iris/src"
  ulimit -c 0   # an aborted multi-rank JAX job dumps hundreds of GB of cores
  cd "$MARIN_ROOT"
}

# The GPU runtime of a training job. XLA command buffers capture the expert-axis NCCL collectives into a CUDA graph
# that hangs the first step on one-GPU-per-rank layouts (marin #5675); NCCL >= 2.29.3 fixes a proxy-slot leak on
# aarch64 (marin #7344).
relay_gpu_env() {
  export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_PGLE=false
  export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${RELAY_XLA_EXTRA:-}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
  export JAX_COMPILATION_CACHE_DIR="$RELAY_ROOT/jax-cache" JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$RELAY_NCCL_IFNAME}
  export NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13 NCCL_WIN_ENABLE=0
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
}
