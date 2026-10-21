# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import optax
import pytest
from haliax.partitioning import set_mesh
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from levanter.checkpoint import save_checkpoint
from levanter.data.text.examples import GrugLmExample
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.import_snowball_hf import (
    ImportSnowballHfConfig,
    snowball_from_hf_state_dict,
    snowball_hf_state_dict,
)
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer
from experiments.june_tpu_67b_a2b.moe.train import (
    GrugTrainState,
    _apply_qb_betas,
    _make_train_step,
    init_weights_only_from_checkpoint,
    initial_state,
)


def test_importer_cli_loads() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "experiments.june_tpu_67b_a2b.moe.import_snowball_hf", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "hf_checkpoint" in result.stdout
    assert "output_path" in result.stdout
    assert "distributed" in result.stdout
    assert "expert_axis_size" in result.stdout
    assert "validate_only" in result.stdout


def test_importer_cli_parses_distributed_geometry() -> None:
    config = draccus.parse(
        ImportSnowballHfConfig,
        args=[
            "--hf_checkpoint",
            "/input",
            "--output_path",
            "/output",
            "--distributed",
            "true",
            "--expert_axis_size",
            "8",
        ],
    )
    assert config.distributed
    assert config.expert_axis_size == 8


def _tiny_model() -> Transformer:
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=4,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        disable_pko=True,
        disable_long_rope=True,
        use_array_stacked_blocks=True,
    )
    return Transformer.init(config, key=jax.random.PRNGKey(0))


def test_hf_round_trip_restores_vendored_layout_without_pending_router_bias() -> None:
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)
    with set_mesh(mesh):
        source = _tiny_model()
        hf_state = snowball_hf_state_dict(source)
        template = eqx.filter_eval_shape(Transformer.init, source.config, key=jax.random.PRNGKey(1))

        imported, pending_qb_betas = snowball_from_hf_state_dict(template, hf_state)

        imported_hf_state = snowball_hf_state_dict(imported)
        assert imported_hf_state.keys() == hf_state.keys()
        for name in hf_state:
            assert jnp.array_equal(imported_hf_state[name], hf_state[name]), name
        assert jnp.array_equal(
            pending_qb_betas,
            jnp.zeros((source.config.num_layers, source.config.num_experts), dtype=jnp.float32),
        )


def test_hf_import_rejects_missing_or_unexpected_tensors() -> None:
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)
    with set_mesh(mesh):
        source = _tiny_model()
        hf_state = snowball_hf_state_dict(source)
        template = eqx.filter_eval_shape(Transformer.init, source.config, key=jax.random.PRNGKey(1))

        missing_state = dict(hf_state)
        missing_state.pop("model.embed_tokens.weight")
        with pytest.raises(ValueError, match=r"missing=.*model\.embed_tokens\.weight"):
            snowball_from_hf_state_dict(template, missing_state)

        unexpected_state = dict(hf_state)
        unexpected_state["unexpected.weight"] = jnp.zeros((1,))
        with pytest.raises(ValueError, match=r"unexpected=.*unexpected\.weight"):
            snowball_from_hf_state_dict(template, unexpected_state)


def test_hf_import_pending_from_router_bias_reproduces_the_base_bias() -> None:
    """--pending_from_router_bias: pending = -bias, so the trainer's -pending (mean-centred) is the base's bias."""
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)
    with set_mesh(mesh):
        source = _tiny_model()
        hf_state = snowball_hf_state_dict(source)
        layers, experts = source.config.num_layers, source.config.num_experts
        bias = jax.random.normal(jax.random.PRNGKey(3), (layers, experts), dtype=jnp.float32)
        bias = bias - jnp.mean(bias, axis=-1, keepdims=True)  # exported biases are mean-centred
        for i in range(layers):
            hf_state[f"model.layers.{i}.mlp.router.bias"] = bias[i]
        template = eqx.filter_eval_shape(Transformer.init, source.config, key=jax.random.PRNGKey(1))

        imported, pending = snowball_from_hf_state_dict(template, hf_state, pending_from_router_bias=True)
        assert pending.dtype == jnp.float32 and pending.shape == (layers, experts)
        assert jnp.allclose(pending, -bias)
        applied = _apply_qb_betas(imported, pending).stacked_blocks.stacked.mlp.router_bias
        assert jnp.allclose(applied, bias, atol=1e-6)
        # the exporter's inverse: router_bias = -pending, mean-centred
        exported = -pending - jnp.mean(-pending, axis=-1, keepdims=True)
        assert jnp.allclose(exported, bias, atol=1e-6)

        _, zero = snowball_from_hf_state_dict(template, hf_state)
        assert not jnp.any(zero)


def test_importer_cli_parses_base_flags() -> None:
    config = draccus.parse(
        ImportSnowballHfConfig,
        args=[
            "--hf_checkpoint",
            "/in",
            "--output_path",
            "/out",
            "--base_config_from_hf",
            "true",
            "--pending_from_router_bias",
            "true",
        ],
    )
    assert config.base_config_from_hf and config.pending_from_router_bias
    default = draccus.parse(ImportSnowballHfConfig, args=["--hf_checkpoint", "/in", "--output_path", "/out"])
    assert not default.base_config_from_hf and not default.pending_from_router_bias


@pytest.mark.parametrize("freeze", [False, True])
def test_train_step_router_bias_freeze(freeze: bool) -> None:
    """freeze_router_bias keeps pending_qb_betas (so the applied router bias) at the init's value; off re-derives it."""
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)
    with set_mesh(mesh):
        params = _tiny_model()
        cfg = params.config
        pending0 = jax.random.normal(jax.random.PRNGKey(5), (cfg.num_layers, cfg.num_experts), dtype=jnp.float32)
        optimizer = optax.sgd(1e-3)
        state = GrugTrainState(
            step=jnp.array(0, dtype=jnp.int32),
            params=params,
            opt_state=optimizer.init(params),
            ema_params=None,
            pending_qb_betas=pending0,
        )
        tokens = jax.random.randint(jax.random.PRNGKey(6), (2, cfg.max_seq_len), 0, cfg.vocab_size)
        batch = GrugLmExample(tokens=tokens, loss_weight=jnp.ones(tokens.shape, jnp.float32))
        step = _make_train_step(
            optimizer,
            jmp.get_policy("params=float32,compute=float32,output=float32"),
            z_loss_weight=0.0,
            ema_beta=None,
            freeze_router_bias=freeze,
        )
        pending_in = jnp.array(pending0)
        state, metrics, _ = step(state, batch)
        state, metrics, _ = step(state, batch)
        if freeze:
            assert jnp.array_equal(state.pending_qb_betas, pending_in)
        else:
            assert jnp.array_equal(state.pending_qb_betas, metrics["qb_beta_per_layer"])
            assert not jnp.array_equal(state.pending_qb_betas, pending_in)


@pytest.mark.parametrize("stored_dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("load_ema", [False, True])
def test_weights_only_import_preserves_small_updates_in_fp32(tmp_path: Path, stored_dtype, load_ema: bool) -> None:
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    optimizer = optax.adam(1e-4)
    with set_mesh(mesh):
        source = jax.tree.map(lambda x: x.astype(stored_dtype) if eqx.is_inexact_array(x) else x, _tiny_model())
        source = eqx.tree_at(lambda m: m.token_embed, source, jnp.ones_like(source.token_embed))
        pending = jnp.full((2, 2), 0.25, dtype=jnp.float32)
        manager = GlobalAsyncCheckpointManager()
        save_checkpoint(
            {"params": source, "pending_qb_betas": pending},
            step=0,
            checkpoint_path=str(tmp_path / "base"),
            manager=manager,
        )
        manager.wait_until_finished()
        fresh = initial_state(
            source.config,
            optimizer=optimizer,
            mp=policy,
            key=jax.random.PRNGKey(1),
            ema_beta=0.9 if load_ema else None,
        )
        loaded = init_weights_only_from_checkpoint(
            fresh,
            str(tmp_path / "base"),
            mesh=mesh,
            load_ema=load_ema,
            mp=policy,
            optimizer=optimizer,
        )
        assert int(loaded.step) == 0
        assert jnp.array_equal(loaded.pending_qb_betas, pending)
        for actual, expected in zip(jax.tree.leaves(loaded.params), jax.tree.leaves(source), strict=True):
            if eqx.is_inexact_array(actual):
                assert actual.dtype == jnp.float32
                assert jnp.array_equal(actual, expected.astype(jnp.float32))
        if load_ema:
            assert eqx.tree_equal(loaded.ema_params, loaded.params)
        else:
            assert loaded.ema_params is None
        assert eqx.tree_equal(loaded.opt_state, optimizer.init(loaded.params))
        params, opt_state = loaded.params, loaded.opt_state
        for _ in range(8):
            grads = jax.tree.map(jnp.ones_like, params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        # Every 1e-4 update rounds away at a BF16 weight of 1. FP32 must retain
        # their accumulated effect, including when the source checkpoint was BF16.
        assert jnp.allclose(params.token_embed, 1.0 - 8e-4, atol=1e-6, rtol=0)
