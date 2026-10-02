# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model, layout and optimizer of relay SFT on Grug 67B-A2B, and the identity of the base it starts from.

The architecture is fixed (the June 67B-A2B MoE). What varies by base is its attention scale and context length:
Grug Datakit 09-21 extends context to 262,144 with ``qk_mult = 1.75``. The importer records the base's
``config.json`` in a ``snowball_base.json`` sidecar inside the native init, and the trainer and the exporter build
the model config from it, so a run can never train or serve at another base's attention temperature.
"""

import dataclasses
import hashlib
import json
import math
from pathlib import Path

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeAdamHConfig

NATIVE_PARAMETERS = 67_078_882_816
MIXED_PRECISION = "params=float32,compute=bfloat16,output=bfloat16"
SEED = 0
Z_LOSS_WEIGHT = 1e-4

# Layout of every published relay SFT model: 16 packed sequences of 65,536 tokens per step on 16 GPUs (4 nodes of
# 4), experts sharded 8 ways, one JAX device per Slurm rank.
SEQUENCE_LENGTH = 65_536
BATCH_SIZE = 16
DEVICES = 16
EXPERT_AXIS_SIZE = 8
MAX_SEGMENTS_PER_PACK = 64

# Optimizer: AdamH, one cosine over every epoch, peak 3e-4 for both parameter groups, 5 % linear warmup.
LEARNING_RATE = 3e-4
WARMUP_FRACTION = 0.05
EPOCHS = 3
MIN_LR_RATIO = 0.1

BASE_SIDECAR = "snowball_base.json"
BOS_ID = 128000

_MODEL = dataclasses.replace(
    MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=65_536),
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
    qk_mult=1.3 * (0.1 * math.log(65_536 / 8_192) + 1.0),
    max_seq_len=SEQUENCE_LENGTH,
    attention_implementation="gpu_fa4_cute",
    ce_implementation="batched_xla",
)

VOCAB_SIZE = _MODEL.vocab_size

# HF config.json key -> model field that must equal the fixed architecture.
_ARCHITECTURE_KEYS = {
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_base_hf_config(hf_dir: str) -> dict:
    """The base checkpoint's ``config.json``, checked against the fixed 67B-A2B architecture."""
    path = Path(hf_dir) / "config.json"
    config = json.loads(path.read_text())
    if config.get("model_type") != "grug_moe":
        raise ValueError(f"{path} is model_type {config.get('model_type')!r}, expected 'grug_moe'.")
    for key, field_name in _ARCHITECTURE_KEYS.items():
        expected = getattr(_MODEL, field_name)
        if config.get(key) != expected:
            raise ValueError(f"{path}: {key}={config.get(key)!r}, but the 67B-A2B model has {field_name}={expected!r}.")
    if int(config.get("head_dim", 0)) != _MODEL.inferred_head_dim:
        raise ValueError(f"{path}: head_dim={config.get('head_dim')!r} != {_MODEL.inferred_head_dim}.")
    if float(config.get("rope_theta", 0.0)) != float(_MODEL.rope.theta):
        raise ValueError(f"{path}: rope_theta={config.get('rope_theta')!r} != {_MODEL.rope.theta}.")
    for key in ("qk_mult", "max_position_embeddings"):
        if key not in config:
            raise ValueError(f"{path} has no {key}; the base's attention scale and context length come from it.")
    return config


def base_sidecar(hf_dir: str) -> dict:
    """The record the importer writes next to a native init: the base's identity and its config.json."""
    config = read_base_hf_config(hf_dir)
    return {
        "hf_base": hf_dir,
        "qk_mult": float(config["qk_mult"]),
        "max_position_embeddings": int(config["max_position_embeddings"]),
        "config_sha256": sha256_file(Path(hf_dir) / "config.json"),
        "tokenizer_sha256": sha256_file(Path(hf_dir) / "tokenizer.json"),
        "pending_qb_betas_from_router_bias": True,
        "config": config,
    }


def read_base_sidecar(init_checkpoint_path: str) -> dict:
    path = Path(init_checkpoint_path) / BASE_SIDECAR
    if not path.is_file():
        raise FileNotFoundError(f"{init_checkpoint_path} has no {BASE_SIDECAR}; import the base with import-hf.")
    sidecar = json.loads(path.read_text())
    if not sidecar.get("pending_qb_betas_from_router_bias"):
        raise ValueError(f"{path}: the init's pending_qb_betas were not set from the base's router bias.")
    return sidecar


def model_config_for_base(base_hf_config: dict, *, max_seq_len: int | None = None) -> GrugModelConfig:
    """The 67B-A2B config with the base's ``qk_mult`` and a context length.

    ``max_seq_len`` is the packing length for training. The importer and the exporter leave it None so the config
    carries the base's ``max_position_embeddings``, which the exported ``config.json`` then repeats.
    """
    return dataclasses.replace(
        _MODEL,
        qk_mult=float(base_hf_config["qk_mult"]),
        max_seq_len=int(max_seq_len if max_seq_len is not None else base_hf_config["max_position_embeddings"]),
    )


def relay_optimizer(*, learning_rate: float, warmup_steps: int) -> GrugMoeAdamHConfig:
    return GrugMoeAdamHConfig(
        learning_rate=learning_rate,
        adam_lr=learning_rate,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
        weight_decay=0.0,
        min_lr_ratio=MIN_LR_RATIO,
        warmup=warmup_steps,
        lr_schedule="cosine",
    )


def warmup_steps_for(schedule_steps: int) -> int:
    """5 % of the schedule, rounded half up, at least 2 (an integer warmup of 1 would read as the fraction 1.0)."""
    return max(2, (schedule_steps * 5 + 50) // 100)
