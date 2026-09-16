#!/bin/bash
#SBATCH --partition=gg
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --job-name=snowball-export-d4

# Native Vista checkpoint -> HF-BF16 export (the vLLM-servable layout), on four Grace CPU
# ranks so the fp32 native params plus the bf16 export never have to fit one host. Mirrors
# vista_snowball_import.sbatch in the other direction. The exporter is Mrinal's
# export_second_cooldown_stage2.py, whose SNOWBALL_EXPORT_* overrides exist for exactly
# this: it initialises JAX from Slurm when SLURM_NTASKS > 1 and writes the Marin chat
# template and the Snowball generation config into the export.
#
#   sbatch -A CCR24067 --export=ALL,MARIN_ROOT=$SCRATCH/marin,MARIN_PYTHON=$SCRATCH/envs/marin-grug-sft/bin/python,\
#     SNOWBALL_EXPORT_CHECKPOINT=<run>/checkpoints/step-75,\
#     SNOWBALL_EXPORT_OUTPUT=<run>/export-step75-hf-bf16,\
#     SNOWBALL_EXPORT_TOKENIZER=<stage-3 HF snapshot dir> \
#     experiments/june_tpu_67b_a2b/moe/vista_snowball_export.sbatch

set -euo pipefail

: "${MARIN_ROOT:?Set MARIN_ROOT to the Marin checkout.}"
: "${MARIN_PYTHON:?Set MARIN_PYTHON to the Levanter Python executable.}"
: "${SNOWBALL_EXPORT_CHECKPOINT:?Set SNOWBALL_EXPORT_CHECKPOINT to the native step-N directory.}"
: "${SNOWBALL_EXPORT_OUTPUT:?Set SNOWBALL_EXPORT_OUTPUT to the new HF export directory.}"
: "${SNOWBALL_EXPORT_TOKENIZER:?Set SNOWBALL_EXPORT_TOKENIZER to the pinned local tokenizer directory.}"

module purge
module load gcc/13.2.0

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
