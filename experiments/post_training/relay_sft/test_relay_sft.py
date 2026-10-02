# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import pytest

from experiments.post_training.relay_sft import recipe
from experiments.post_training.relay_sft.prerendered import ROW_ADAPTER
from experiments.post_training.relay_sft.relay_sft import CACHE_MANIFEST, CacheManifest, relay_run_config
from experiments.post_training.relay_sft.rows import PackCount

EPOCH_STEPS = 401


@pytest.fixture
def inputs(tmp_path):
    tokenizer = tmp_path / "base"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"model": "base"}')
    tokenizer_sha256 = recipe.sha256_file(tokenizer / "tokenizer.json")

    init = tmp_path / "init" / "step-0"
    init.mkdir(parents=True)
    (init / "metadata.json").write_text(json.dumps({"step": 0, "timestamp": "2026-10-01T00:00:00"}))
    sidecar = {
        "hf_base": str(tokenizer),
        "tokenizer_sha256": tokenizer_sha256,
        "pending_qb_betas_from_router_bias": True,
        "config": {"qk_mult": 1.75, "max_position_embeddings": 262_144},
    }
    (init / recipe.BASE_SIDECAR).write_text(json.dumps(sidecar))

    cache = tmp_path / "cache"
    (cache / "train").mkdir(parents=True)
    packs = PackCount(rows=10_308, tokens=324_165_185, packs=6_410, epoch_steps=EPOCH_STEPS)
    manifest = CacheManifest(
        parquet="/data/train-00000-of-00001.parquet",
        parquet_sha256="0" * 64,
        tokenizer_sha256=tokenizer_sha256,
        sequence_length=recipe.SEQUENCE_LENGTH,
        batch_size=recipe.BATCH_SIZE,
        row_adapter=ROW_ADAPTER,
        packs=packs,
    )
    (cache / CACHE_MANIFEST).write_text(json.dumps(dataclasses.asdict(manifest)))
    (cache / "train" / ".stats.json").write_text(
        json.dumps({"total_elements": packs.rows, "total_tokens": packs.tokens})
    )
    return {
        "init_checkpoint_path": str(init),
        "cache_path": str(cache),
        "tokenizer_path": str(tokenizer),
        "output_path": str(tmp_path / "run" / "train"),
        "run_id": "relay-test",
        "epochs": 3,
        "learning_rate": 3e-4,
        "steps": None,
        "resume": False,
    }


def test_run_trains_every_epoch_on_one_cosine_with_the_router_bias_frozen(inputs):
    config = relay_run_config(**inputs)

    assert config.trainer.trainer.num_train_steps == 3 * EPOCH_STEPS
    assert config.trainer.schedule_steps == 3 * EPOCH_STEPS
    assert config.optimizer.warmup == 60
    assert config.optimizer.learning_rate == config.optimizer.adam_lr == 3e-4
    assert config.trainer.trainer.checkpointer.keep == [{"every": EPOCH_STEPS}]
    assert config.trainer.freeze_router_bias
    assert config.model.qk_mult == 1.75
    assert config.model.max_seq_len == recipe.SEQUENCE_LENGTH
    assert config.trainer.trainer.initialize_from == inputs["init_checkpoint_path"]


def test_steps_trains_the_start_of_the_full_schedule(inputs):
    config = relay_run_config(**{**inputs, "steps": 30})

    assert config.trainer.trainer.num_train_steps == 30
    assert config.trainer.schedule_steps == 3 * EPOCH_STEPS
    assert config.optimizer.warmup == 60


def test_cache_from_other_rows_is_refused(inputs):
    stats = f"{inputs['cache_path']}/train/.stats.json"
    with open(stats, "w") as handle:
        json.dump({"total_elements": 10, "total_tokens": 1000}, handle)

    with pytest.raises(ValueError, match="incomplete or from other rows"):
        relay_run_config(**inputs)


def test_tokenizer_other_than_the_base_is_refused(inputs, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "tokenizer.json").write_text('{"model": "other"}')

    with pytest.raises(ValueError, match="tokenizer"):
        relay_run_config(**{**inputs, "tokenizer_path": str(other)})


def test_existing_output_needs_resume(inputs, tmp_path):
    (tmp_path / "run" / "train").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="--resume"):
        relay_run_config(**inputs)
