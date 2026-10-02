# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer
from experiments.post_training.relay_sft.hf_conversion import (
    changed_leaves,
    hf_state_dict,
    leaf_abs_sums,
    native_from_hf_state_dict,
    with_router_bias_from_pending,
)


@pytest.fixture
def mesh():
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)
    with set_mesh(mesh):
        yield mesh


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


def _template(model: Transformer) -> Transformer:
    return eqx.filter_eval_shape(Transformer.init, model.config, key=jax.random.PRNGKey(1))


def _hf_state_with_bias(model: Transformer) -> tuple[dict, jax.Array]:
    layers, experts = model.config.num_layers, model.config.num_experts
    bias = jax.random.normal(jax.random.PRNGKey(3), (layers, experts), dtype=jnp.float32)
    bias = bias - jnp.mean(bias, axis=-1, keepdims=True)  # exported biases are mean-centred
    state = hf_state_dict(model)
    for layer in range(layers):
        state[f"model.layers.{layer}.mlp.router.bias"] = bias[layer]
    return state, bias


def test_import_then_export_reproduces_every_hf_tensor(mesh):
    state, _ = _hf_state_with_bias(_tiny_model())

    imported, pending = native_from_hf_state_dict(_template(_tiny_model()), state)
    exported = hf_state_dict(with_router_bias_from_pending(imported, pending))

    assert exported.keys() == state.keys()
    for name in state:
        assert jnp.allclose(exported[name], state[name], atol=1e-6), name


def test_import_sets_pending_to_minus_the_base_router_bias(mesh):
    state, bias = _hf_state_with_bias(_tiny_model())

    _, pending = native_from_hf_state_dict(_template(_tiny_model()), state)

    assert pending.dtype == jnp.float32
    assert jnp.allclose(pending, -bias)


def test_import_rejects_missing_or_unexpected_tensors(mesh):
    state, _ = _hf_state_with_bias(_tiny_model())
    template = _template(_tiny_model())

    with pytest.raises(ValueError, match=r"missing=.*model\.embed_tokens\.weight"):
        native_from_hf_state_dict(template, {k: v for k, v in state.items() if k != "model.embed_tokens.weight"})
    with pytest.raises(ValueError, match=r"unexpected=.*extra\.weight"):
        native_from_hf_state_dict(template, {**state, "extra.weight": jnp.zeros((1,))})


def test_changed_leaves_flags_a_lost_chunk_but_not_reduction_noise():
    tree = {"experts": jnp.full((4, 128), 0.5), "bias": jnp.zeros((4,))}
    before = leaf_abs_sums(tree)
    assert before == {"['bias']": 0.0, "['experts']": 256.0}
    noisy = {"['bias']": 0.0, "['experts']": 256.0 * (1 + 1e-6)}
    assert changed_leaves(before, noisy) == []
    lost = leaf_abs_sums({"experts": tree["experts"].at[0].set(0.0), "bias": tree["bias"]})
    assert changed_leaves(before, lost) == ["['experts']"]
    assert changed_leaves(before, {"['bias']": 0.0}) == ["['experts']"]
