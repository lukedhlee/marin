#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Build the GPU env the relay SFT jobs run in: uv sync from this checkout's lock, NCCL raised to 2.30.7 (the lock's
# 2.28.9 leaks proxy-op slots on aarch64 and wedges, marin #7344), then an import smoke.
#
#   Jupiter (login nodes have internet; keep the uv cache off fscratch):
#     RELAY_CLUSTER=jupiter RELAY_ACCOUNT=<project> RELAY_ROOT=/e/data1/<group>/$USER/relay-sft bash build_env.sh
#   Horizon (installs run on a compute node, which reaches the internet through a login-side SOCKS tunnel,
#   OpenThoughts-Agent data/r2egym/horizon/tunnel.sh <jobid> 18080):
#     sbatch -A <allocation> -p debug -N 1 -t 01:00:00 --export=ALL,RELAY_CLUSTER=horizon,RELAY_PROXY_PORT=18080,... build_env.sh
set -euo pipefail
: "${MARIN_ROOT:=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
source "$MARIN_ROOT/experiments/post_training/relay_sft/slurm/cluster.sh"
NCCL_VERSION=${NCCL_VERSION:-2.30.7}
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
ENV_DIR=$(dirname "$(dirname "$RELAY_PYTHON")")

if [ -n "${RELAY_PROXY_PORT:-}" ]; then
  for _ in $(seq 120); do (echo > "/dev/tcp/127.0.0.1/$RELAY_PROXY_PORT") 2>/dev/null && break; sleep 5; done
  export ALL_PROXY=socks5h://127.0.0.1:$RELAY_PROXY_PORT HTTPS_PROXY=socks5h://127.0.0.1:$RELAY_PROXY_PORT
  export NO_PROXY=localhost,127.0.0.1
fi

relay_compute_env
export UV_CACHE_DIR=${UV_CACHE_DIR:-$RELAY_ROOT/cache/uv} UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=$ENV_DIR
export UV_CONCURRENT_DOWNLOADS=8 UV_CONCURRENT_INSTALLS=8 UV_CONCURRENT_BUILDS=2
mkdir -p "$UV_CACHE_DIR"
echo "ENV_BUILD_START env=$ENV_DIR commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
[ -x "$RELAY_PYTHON" ] || "$UV" venv --python 3.12 "$ENV_DIR"
"$UV" sync --all-packages --extra=gpu --frozen
"$UV" pip install --python "$RELAY_PYTHON" --no-deps "nvidia-nccl-cu13==$NCCL_VERSION"
unset ALL_PROXY HTTPS_PROXY

JAX_PLATFORMS=cpu "$RELAY_PYTHON" - <<'PY'
import ctypes, glob, os, sys
libs = sorted(glob.glob(os.path.join(sys.prefix, "**", "libnccl.so.2"), recursive=True))
version = ctypes.c_int()
ctypes.CDLL(libs[0]).ncclGetVersion(ctypes.byref(version))
major, rest = divmod(version.value, 10000)
minor, patch = divmod(rest, 100)
assert (major, minor, patch) >= (2, 29, 3), f"NCCL {major}.{minor}.{patch} is below 2.29.3 (marin #7344)"
import jax
import experiments.post_training.relay_sft.relay_sft  # noqa: F401
print(f"nccl={major}.{minor}.{patch} jax={jax.__version__}")
PY
echo "ENV_BUILD_OK $(date -u +%FT%TZ)"
