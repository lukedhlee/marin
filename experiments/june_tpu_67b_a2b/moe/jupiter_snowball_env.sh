#!/bin/bash
# Build the Levanter GPU env for the Snowball SFT on a Jupiter (JSC) LOGIN node.
#
# Compute nodes have no internet, so the `step_env` of vista_r2egym_prep.sbatch moves here: uv sync from
# this checkout's workspace lock (the same JAX 0.11 / FA4 / cuTe pins Mrinal's completed stages ran on),
# then the NCCL override to a release with the aarch64 proxy-slot fix (marin #7344 -> >= 2.29.3), then a
# CPU import smoke. The GPU-side checks run in jupiter_r2egym_prep.sbatch.
#
# Login-node limits (see .claude/ops/jupiter/ops.md): 4,096 pids including threads per user, no
# torch/Ray; uv's concurrency is capped below. Keep the uv cache OFF fscratch (inode-heavy; the reformo
# fscratch quota is shared): SNOWBALL_UV_CACHE=/e/data1/mmlaion/$USER/cache/uv-grug.
#
#   MARIN_ROOT=<checkout> SNOWBALL_ENV=<env dir> SNOWBALL_UV_CACHE=<cache dir> bash jupiter_snowball_env.sh
#
# Idempotent: an existing env is reused (sync is a no-op), the NCCL override is re-applied, the smoke re-run.
set -euo pipefail
: "${MARIN_ROOT:?}"; : "${SNOWBALL_ENV:?}"; : "${SNOWBALL_UV_CACHE:?}"
NCCL_OVERRIDE="${NCCL_OVERRIDE:-2.30.7}"   # the release Mrinal's completed stages ran on
UV="${UV:-$HOME/.local/bin/uv}"

# Non-interactive ssh shells do not define `module`; load Lmod the way build_snowball_env.sh does.
type module >/dev/null 2>&1 || source "${LMOD_PKG:-/e/software/default/lmod/8.7.64}/init/bash"
module purge 2>/dev/null; module load Stages/2026 GCC/14.3.0 CUDA/13
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
export UV_CACHE_DIR="${SNOWBALL_UV_CACHE}" UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="${SNOWBALL_ENV}"
export UV_CONCURRENT_DOWNLOADS=8 UV_CONCURRENT_INSTALLS=8 UV_CONCURRENT_BUILDS=1
export OMP_NUM_THREADS=1 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${MARIN_ROOT}:${MARIN_ROOT}/lib/levanter/src:${MARIN_ROOT}/lib/haliax/src:${MARIN_ROOT}/lib/marin/src:${MARIN_ROOT}/lib/rigging/src:${MARIN_ROOT}/lib/fray/src:${MARIN_ROOT}/lib/zephyr/src:${MARIN_ROOT}/lib/iris/src"
PY="${SNOWBALL_ENV}/bin/python"
mkdir -p "${UV_CACHE_DIR}"
cd "${MARIN_ROOT}"
echo "ENV_BUILD_START host=$(hostname) env=${SNOWBALL_ENV} nccl_override=${NCCL_OVERRIDE} commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
"${UV}" --version

[ -x "${PY}" ] || "${UV}" venv --python 3.12 "${SNOWBALL_ENV}"
# The workspace lock, unchanged: the same JAX/FA4/cuTe pins Mrinal's completed stages ran on.
"${UV}" sync --all-packages --extra=gpu --frozen
# marin #7344: NCCL 2.28.9 (the lock's pin) leaks proxy-op slots on aarch64 and wedges silently.
"${UV}" pip install --python "${PY}" --no-deps "nvidia-nccl-cu13==${NCCL_OVERRIDE}"

JAX_PLATFORMS=cpu "${PY}" - <<'PYEOF'
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
echo "ENV_BUILD_OK $(date -u +%FT%TZ)"
