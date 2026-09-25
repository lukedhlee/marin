# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug Datakit 09-21 SFT path: base config plumbing, the Datakit row mapping, and the Stage-3 path unchanged."""

import dataclasses
import hashlib
import json

import pytest

from experiments.june_tpu_67b_a2b.moe.grug_datakit_chat import (
    GRUG_DATAKIT_0921_TOKENIZER_SHA256,
    GRUG_DATAKIT_0921_TRAINING_TEMPLATE,
    GRUG_DATAKIT_0921_TRAINING_TEMPLATE_SHA256,
    SnowballDatakitChatFormat,
    datakit_row_to_chat,
)
from experiments.june_tpu_67b_a2b.moe.snowball_chat_recipe import (
    SNOWBALL_BASE_SIDECAR,
    SNOWBALL_CHAT_MODEL_CONFIG,
    read_base_hf_config,
    read_init_base_sidecar,
    snowball_model_config_for_base,
)
from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import (
    SNOWBALL_TOKENIZER_SHA256,
    STAGES,
    _format_identity,
    format_for_stage,
    snowball_chat_format,
    validate_init_base,
)

# config.json of grug-datakit-sft-20260921 (the fields the recipe checks + the two that differ from Stage-3)
BASE_0921 = {
    "model_type": "grug_moe", "vocab_size": 128256, "hidden_size": 2560, "num_hidden_layers": 26,
    "num_attention_heads": 20, "num_key_value_heads": 5, "head_dim": 128, "max_position_embeddings": 262144,
    "sliding_window": 2048, "rope_theta": 10000.0, "num_experts": 256, "num_experts_per_tok": 4,
    "moe_intermediate_size": 1280, "shared_expert_intermediate_size": 2560, "qk_mult": 1.75,
}
BESPOKE = ("bespoke_fold_all", "bespoke_fold_noglm", "bespoke_think_all", "bespoke_think_noglm")


def test_template_is_the_pinned_0921_training_template() -> None:
    assert hashlib.sha256(GRUG_DATAKIT_0921_TRAINING_TEMPLATE.encode()).hexdigest() == (
        GRUG_DATAKIT_0921_TRAINING_TEMPLATE_SHA256
    )
    assert "{% generation %}" in GRUG_DATAKIT_0921_TRAINING_TEMPLATE


def test_base_config_carries_qk_mult_and_context(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(BASE_0921))
    cfg = read_base_hf_config(str(tmp_path))
    model = snowball_model_config_for_base(cfg)
    assert model.qk_mult == 1.75 and model.max_seq_len == 262144
    train = snowball_model_config_for_base(cfg, max_seq_len=65536)
    assert train.qk_mult == 1.75 and train.max_seq_len == 65536
    # everything else is the recipe
    assert dataclasses.replace(model, qk_mult=SNOWBALL_CHAT_MODEL_CONFIG.qk_mult,
                               max_seq_len=SNOWBALL_CHAT_MODEL_CONFIG.max_seq_len) == SNOWBALL_CHAT_MODEL_CONFIG
    assert snowball_model_config_for_base(None) is SNOWBALL_CHAT_MODEL_CONFIG


def test_base_config_rejects_a_different_architecture(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({**BASE_0921, "num_experts": 128}))
    with pytest.raises(ValueError, match="num_experts"):
        read_base_hf_config(str(tmp_path))


def test_row_mapping() -> None:
    row = {
        "id": "r1",
        "enable_thinking": False,
        "conversations": [
            {"role": "user", "content": "do it", "reasoning": None},
            {"role": "assistant", "content": "{}", "reasoning": ""},
            {"role": "user", "content": "more", "reasoning": None},
            {"role": "assistant", "content": "{\"a\": 1}", "reasoning": "think hard"},
        ],
    }
    out = datakit_row_to_chat(row, messages_field="conversations")
    assert out["chat_template_kwargs"] == {"enable_thinking": False}
    assert out["conversations"][1] == {"role": "assistant", "content": "{}"}
    assert out["conversations"][3]["reasoning_content"] == "think hard"
    assert all("reasoning" not in m for m in out["conversations"])
    with pytest.raises(ValueError, match="enable_thinking"):
        datakit_row_to_chat({**row, "enable_thinking": None}, messages_field="conversations")


def test_bespoke_stages() -> None:
    for name in BESPOKE:
        spec = STAGES[name]
        assert spec.datakit_format and spec.freeze_router_bias and spec.requires_base_sidecar
        assert spec.tokenizer_sha256 == GRUG_DATAKIT_0921_TOKENIZER_SHA256
        assert spec.dataset_id == "private/bespoke-qwen-glm-successful-20260922"
        fmt = format_for_stage(name)
        assert isinstance(fmt, SnowballDatakitChatFormat) and fmt.messages_field == "conversations"
        assert _format_identity(fmt)["row_adapter"] == "datakit_reasoning_enable_thinking_v1"


def test_existing_stages_are_unchanged() -> None:
    for name, spec in STAGES.items():
        if name in BESPOKE:
            continue
        assert not spec.datakit_format and not spec.requires_base_sidecar
        # Ben's recipe stages keep the per-batch bias; every SFT from an HF import freezes it (2026-09-24).
        assert spec.freeze_router_bias == (name not in ("chat", "thinking", "nemotron_terminal"))
        assert spec.tokenizer_sha256 == SNOWBALL_TOKENIZER_SHA256 and spec.dataset_id is None
        fmt = format_for_stage(name)
        assert fmt == snowball_chat_format(messages_field=spec.messages_field, chat_template=spec.chat_template)
        assert set(_format_identity(fmt)) == {"messages_field", "mask_user_turns", "chat_template_sha256"}


def test_init_base_sidecar_gate(tmp_path, monkeypatch) -> None:
    stage3_init = tmp_path / "s3"
    stage3_init.mkdir()
    assert read_init_base_sidecar(str(stage3_init)) is None
    # A Stage-3 stage freezes the router bias by default, so a sidecar-less (zeroed-bias) init is refused
    # unless the run opts out of the freeze.
    with pytest.raises(ValueError, match="base_config_from_hf"):
        validate_init_base(str(stage3_init), "ota3_if")
    monkeypatch.setenv("SNOWBALL_FREEZE_ROUTER_BIAS", "0")
    assert validate_init_base(str(stage3_init), "ota3_if") is None
    monkeypatch.delenv("SNOWBALL_FREEZE_ROUTER_BIAS")
    with pytest.raises(ValueError, match="base_config_from_hf"):
        validate_init_base(str(stage3_init), "bespoke_think_all")
    stage3_frozen = tmp_path / "s3_frozen"
    stage3_frozen.mkdir()
    (stage3_frozen / SNOWBALL_BASE_SIDECAR).write_text(
        json.dumps({"config": {}, "tokenizer_sha256": SNOWBALL_TOKENIZER_SHA256, "pending_qb_betas_from_router_bias": True})
    )
    assert validate_init_base(str(stage3_frozen), "ota3_if") is not None

    init = tmp_path / "dk0921"
    init.mkdir()
    record = {"config": BASE_0921, "qk_mult": 1.75, "tokenizer_sha256": GRUG_DATAKIT_0921_TOKENIZER_SHA256,
              "pending_qb_betas_from_router_bias": True}
    (init / SNOWBALL_BASE_SIDECAR).write_text(json.dumps(record))
    assert validate_init_base(str(init), "bespoke_think_all")["qk_mult"] == 1.75
    with pytest.raises(ValueError, match="tokenizer"):
        validate_init_base(str(init), "ota3_if")  # a Stage-3 stage refuses the 09-21 init
    (init / SNOWBALL_BASE_SIDECAR).write_text(json.dumps({**record, "pending_qb_betas_from_router_bias": False}))
    with pytest.raises(ValueError, match="zeroed"):
        validate_init_base(str(init), "bespoke_think_all")
