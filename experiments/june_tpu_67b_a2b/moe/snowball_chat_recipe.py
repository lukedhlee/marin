# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball Base-to-Chat model and optimizer contract."""

import dataclasses
import json
import math
import os
from pathlib import Path

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeAdamHConfig

# Packing length, sequences per step and device count. The env knobs exist for the small-batch Grug Datakit 09-21
# arms (16 x 65,536 on 4 Jupiter nodes); unset, every value is the Stage-3 contract. They are read at import, so
# every process of a run (launcher, preflight, the srun ranks) must see the same environment (sbatch --export=ALL).
SNOWBALL_CHAT_SEQUENCE_LENGTH = int(os.environ.get("SNOWBALL_SEQ_LEN") or 32_768)
SNOWBALL_CHAT_BATCH_SIZE = int(os.environ.get("SNOWBALL_BATCH") or 64)
SNOWBALL_CHAT_STEPS = 257
SNOWBALL_CHAT_TOKENS = 538_877_811
SNOWBALL_CHAT_EXAMPLES = 385_700
SNOWBALL_CHAT_DEVICES = int(os.environ.get("SNOWBALL_DEVICES") or 64)
SNOWBALL_CHAT_EXPERT_PARALLEL = 8
SNOWBALL_CHAT_REPLICA_AXIS = 1
SNOWBALL_CHAT_MODEL_AXIS = 1
SNOWBALL_CHAT_SEED = 0
SNOWBALL_CHAT_MP = "params=float32,compute=bfloat16,output=bfloat16"
SNOWBALL_NATIVE_PARAMETERS = 67_078_882_816

_QK_MULT = 1.3 * (0.1 * math.log(65_536 / 8_192) + 1.0)
_MODEL_BASE = MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=65_536)

SNOWBALL_CHAT_MODEL_CONFIG = dataclasses.replace(
    _MODEL_BASE,
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
    qk_mult=_QK_MULT,
    max_seq_len=SNOWBALL_CHAT_SEQUENCE_LENGTH,
    attention_implementation="gpu_fa4_cute",
    ce_implementation="batched_xla",
)

# Base-checkpoint identity. The recipe above is Stage-3's (qk_mult = the YaRN scale for 65,536 / 8,192). A base
# with a different context extension (Grug Datakit 09-21: qk_mult 1.75, max_position_embeddings 262,144) must carry
# ITS values into import, train and export, or the model trains and serves at the wrong attention temperature
# without any error. ``import_snowball_hf`` writes this sidecar into the native init it builds from such a base;
# the trainer and the exporter read the model config from it (or from the base's config.json) instead of the
# constant. An init without the sidecar is a Stage-3 init and keeps SNOWBALL_CHAT_MODEL_CONFIG byte for byte.
SNOWBALL_BASE_SIDECAR = "snowball_base.json"

# HF config.json key -> GrugModelConfig field that must equal the recipe (the architecture itself is fixed).
_BASE_ARCH_KEYS = {
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_dim",
    "num_hidden_layers": "num_layers",
    "num_attention_heads": "num_heads",
    "num_key_value_heads": "num_kv_heads",
    "sliding_window": "sliding_window",
    "num_experts": "num_experts",
    "num_experts_per_tok": "num_experts_per_token",
    "moe_intermediate_size": "intermediate_dim",
    "shared_expert_intermediate_size": "shared_expert_intermediate_dim",
}


def read_base_hf_config(hf_dir: str) -> dict:
    """The base checkpoint's HF config.json, checked against the fixed Snowball architecture."""
    path = Path(hf_dir) / "config.json"
    cfg = json.loads(path.read_text())
    if cfg.get("model_type") != "grug_moe":
        raise ValueError(f"{path} is model_type {cfg.get('model_type')!r}, expected 'grug_moe'.")
    for key, field_name in _BASE_ARCH_KEYS.items():
        want = getattr(SNOWBALL_CHAT_MODEL_CONFIG, field_name)
        if cfg.get(key) != want:
            raise ValueError(f"{path}: {key}={cfg.get(key)!r}, but the Snowball recipe has {field_name}={want!r}.")
    if int(cfg.get("head_dim", 0)) != SNOWBALL_CHAT_MODEL_CONFIG.inferred_head_dim:
        raise ValueError(f"{path}: head_dim={cfg.get('head_dim')!r} != {SNOWBALL_CHAT_MODEL_CONFIG.inferred_head_dim}.")
    if float(cfg.get("rope_theta", 0.0)) != float(SNOWBALL_CHAT_MODEL_CONFIG.rope.theta):
        raise ValueError(f"{path}: rope_theta={cfg.get('rope_theta')!r} != {SNOWBALL_CHAT_MODEL_CONFIG.rope.theta}.")
    for key in ("qk_mult", "max_position_embeddings"):
        if key not in cfg:
            raise ValueError(f"{path} has no {key}; cannot take the base's attention scale / context length from it.")
    return cfg


def snowball_model_config_for_base(base_hf_config: dict | None, *, max_seq_len: int | None = None):
    """The recipe's model config with the base's qk_mult (and, unless given, its max_position_embeddings).

    ``None`` returns SNOWBALL_CHAT_MODEL_CONFIG unchanged (the Stage-3 path). ``max_seq_len`` is the training
    packing length for the trainer; the importer and the exporter leave it None so the config carries the base's
    context length (the exported config.json's max_position_embeddings).
    """
    if base_hf_config is None:
        return SNOWBALL_CHAT_MODEL_CONFIG
    return dataclasses.replace(
        SNOWBALL_CHAT_MODEL_CONFIG,
        qk_mult=float(base_hf_config["qk_mult"]),
        max_seq_len=int(max_seq_len if max_seq_len is not None else base_hf_config["max_position_embeddings"]),
    )


def read_init_base_sidecar(init_checkpoint_path: str) -> dict | None:
    """The sidecar ``import_snowball_hf`` wrote into a non-Stage-3 init, or None (a Stage-3 init)."""
    path = Path(init_checkpoint_path) / SNOWBALL_BASE_SIDECAR
    return json.loads(path.read_text()) if path.is_file() else None


SNOWBALL_CHAT_OPTIMIZER = GrugMoeAdamHConfig(
    learning_rate=5e-5,
    adam_lr=5e-5,
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    max_grad_norm=1.0,
    weight_decay=0.0,
    min_lr_ratio=0.1,
    warmup=0.03,
    lr_schedule="cosine",
)
