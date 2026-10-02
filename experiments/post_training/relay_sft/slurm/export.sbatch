#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Native checkpoint (CHECKPOINT, a step-N dir) -> bf16 HF export (EXPORT) carrying the base's (BASE) config values,
# serving and training templates and tokenizer files. Four CPU ranks, one per node, as import.sbatch.
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00
#SBATCH --job-name=relay-sft-export
set -euo pipefail
: "${MARIN_ROOT:?}" "${BASE:?}" "${CHECKPOINT:?}" "${EXPORT:?}"
source "$MARIN_ROOT/experiments/post_training/relay_sft/slurm/cluster.sh"
relay_compute_env
export JAX_PLATFORMS=cpu
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task="$RELAY_NODE_CPUS" --cpu-bind=cores \
  "$RELAY_PYTHON" -m experiments.post_training.relay_sft.relay_sft export-hf --checkpoint "$CHECKPOINT" --base "$BASE" \
  --output-path "$EXPORT"
"$RELAY_PYTHON" - "$EXPORT" "$BASE" <<'PY'
import json, os, sys
export, base = sys.argv[1:]
config, base_config = (json.load(open(os.path.join(d, "config.json"))) for d in (export, base))
assert config["model_type"] == "grug_moe" and config["architectures"] == ["GrugMoeForCausalLM"], config
for key in ("qk_mult", "max_position_embeddings"):
    assert config[key] == base_config[key], f"export {key}={config[key]} != base {base_config[key]}"
assert any(name.endswith(".safetensors") for name in os.listdir(export)), "no safetensors shards"
print(f"EXPORT_CHECK_OK qk_mult={config['qk_mult']} max_position_embeddings={config['max_position_embeddings']}")
PY
