# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""freeze_router_bias keeps the router bias at the init's value; off, each step re-derives it from the batch."""

import jax
import jax.numpy as jnp
import jmp
import optax
import pytest
from haliax.partitioning import set_mesh
from levanter.data.text.examples import GrugLmExample
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer
from experiments.june_tpu_67b_a2b.moe.train import GrugTrainState, _make_train_step


@pytest.mark.parametrize("freeze", [False, True])
def test_train_step_router_bias(freeze: bool):
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
    with set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)):
        params = Transformer.init(config, key=jax.random.PRNGKey(0))
        initial_pending = jax.random.normal(jax.random.PRNGKey(5), (config.num_layers, config.num_experts))
        optimizer = optax.sgd(1e-3)
        state = GrugTrainState(
            step=jnp.array(0, dtype=jnp.int32),
            params=params,
            opt_state=optimizer.init(params),
            ema_params=None,
            pending_qb_betas=jnp.array(initial_pending),
        )
        tokens = jax.random.randint(jax.random.PRNGKey(6), (2, config.max_seq_len), 0, config.vocab_size)
        batch = GrugLmExample(tokens=tokens, loss_weight=jnp.ones(tokens.shape, jnp.float32))
        step = _make_train_step(
            optimizer,
            jmp.get_policy("params=float32,compute=float32,output=float32"),
            z_loss_weight=0.0,
            ema_beta=None,
            freeze_router_bias=freeze,
        )

        state, metrics, _ = step(state, batch)
        state, metrics, _ = step(state, batch)

    if freeze:
        assert jnp.array_equal(state.pending_qb_betas, initial_pending)
    else:
        assert jnp.array_equal(state.pending_qb_betas, metrics["qb_beta_per_layer"])
        assert not jnp.array_equal(state.pending_qb_betas, initial_pending)
