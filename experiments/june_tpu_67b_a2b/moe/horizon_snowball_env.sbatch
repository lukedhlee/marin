#!/bin/bash
# Build the Levanter GPU env for the Snowball SFT on ONE TACC Horizon compute node (4x GB200, aarch64, sm_100).
#
# Horizon port of jupiter_snowball_env.sh. Same lock, same steps: uv sync from this checkout's workspace lock
# (JAX 0.11 / FA4 / cuTe pins), the NCCL override to 2.30.7 (marin #7344), a CPU import smoke. Two deltas:
#   - it runs on a compute node (Horizon's login node is shared; TACC policy keeps installs off it), and
#     compute nodes have no internet, so it waits for OT-Agent's data/r2egym/horizon/tunnel.sh <jobid> on the
#     login node and goes out through that SOCKS proxy;
#   - it adds a GPU smoke (JAX sees the four GB200s) and prints the node's network devices, which the
#     multi-node scripts need for NCCL_SOCKET_IFNAME.
# Compiler: TACC's login env loads nvidia/26.9 (CC=nvc); every Horizon job uses the system GCC 14 instead.
#
#   sbatch --export=ALL,MARIN_ROOT=<checkout>,SNOWBALL_ENV=<env dir>,SNOWBALL_UV_CACHE=<cache dir> horizon_snowball_env.sbatch
#   setsid nohup bash <OT-Agent>/data/r2egym/horizon/tunnel.sh <jobid> > ~/tunnel.<jobid>.log 2>&1 &
#SBATCH -J snowball-sft-env
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 01:00:00
set -uo pipefail
: "${MARIN_ROOT:?}"; : "${SNOWBALL_ENV:?}"; : "${SNOWBALL_UV_CACHE:?}"
NCCL_OVERRIDE="${NCCL_OVERRIDE:-2.30.7}"   # the release Mrinal's completed stages and the Jupiter runs used
UV="${UV:-$HOME/snowball/bin/uv}"
PORT="${PORT:-18080}"

echo "ENV_JOB host=$(hostname) start $(date -u +%FT%TZ); waiting for the reverse tunnel on :$PORT"
for i in $(seq 1 120); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 5; done
(echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "FATAL: no tunnel on :$PORT after 10 min (run tunnel.sh on login1)"; exit 1; }
P=socks5h://127.0.0.1:$PORT
export ALL_PROXY=$P all_proxy=$P HTTPS_PROXY=$P https_proxy=$P HTTP_PROXY=$P http_proxy=$P NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
curl -sS -m 20 -o /dev/null -w "preflight pypi %{http_code}\n" https://pypi.org/simple/ || { echo "FATAL: proxy does not reach pypi"; exit 1; }

echo "--- node network (for NCCL_SOCKET_IFNAME) ---"; ip -br a; ls /sys/class/infiniband 2>/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi topo -m 2>/dev/null | head -12

export CC=gcc CXX=g++
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
export UV_CACHE_DIR="${SNOWBALL_UV_CACHE}" UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="${SNOWBALL_ENV}"
export UV_CONCURRENT_DOWNLOADS=16 UV_CONCURRENT_INSTALLS=16 UV_CONCURRENT_BUILDS=4
export OMP_NUM_THREADS=1 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src"
PY="${SNOWBALL_ENV}/bin/python"
mkdir -p "${UV_CACHE_DIR}"
cd "${MARIN_ROOT}"
echo "ENV_BUILD_START env=${SNOWBALL_ENV} nccl_override=${NCCL_OVERRIDE} commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
"${UV}" --version

[ -x "${PY}" ] || "${UV}" venv --python 3.12 "${SNOWBALL_ENV}" || exit 1
"${UV}" sync --all-packages --extra=gpu --frozen || exit 1
"${UV}" pip install --python "${PY}" --no-deps "nvidia-nccl-cu13==${NCCL_OVERRIDE}" || exit 1

JAX_PLATFORMS=cpu "${PY}" - <<'PYEOF' || exit 1
import ctypes, glob, os, sys
sos = sorted(glob.glob(os.path.join(sys.prefix, "**", "libnccl.so.2"), recursive=True))
assert sos, "no libnccl.so.2 in the env"
lib = ctypes.CDLL(sos[0]); v = ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(v))
major, rest = divmod(v.value, 10000); minor, patch = divmod(rest, 100)
print(f"nccl_runtime={major}.{minor}.{patch} ({sos[0]})")
assert (major, minor, patch) >= (2, 29, 3), "NCCL below 2.29.3 -- marin #7344"
import jax; print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())
import haliax, levanter; print("haliax + levanter import OK")
import experiments.june_tpu_67b_a2b.moe.vista_snowball_chat as launcher
print("launcher stages", sorted(launcher.STAGES))
PYEOF
unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy
JAX_PLATFORMS=cuda "${PY}" - <<'PYEOF' || exit 1
import jax, jax.numpy as jnp
d = jax.devices(); print("gpu devices", d)
assert len(d) == 4 and d[0].platform == "gpu", d
x = jnp.ones((4096, 4096), jnp.bfloat16); print("matmul ok", float((x @ x)[0, 0]))
PYEOF
echo "ENV_BUILD_OK $(date -u +%FT%TZ)"
