# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert between a Grug 67B-A2B HF export and the June trainer's native checkpoint.

The trainer holds the model as ``ArrayStacked`` blocks plus a ``pending_qb_betas`` router state, from which each
step sets the router bias (``-pending``, mean-centred per layer). An HF export holds one tensor group per layer with
the bias already applied. The import reverses the layout and sets ``pending_qb_betas = -router_bias``, so the first
step routes with the base's own bias; with ``freeze_router_bias`` it stays there. The export writes
``router_bias = -pending`` mean-centred, the inverse.
"""

import dataclasses
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from levanter.checkpoint import load_checkpoint
from levanter.tokenizers import load_tokenizer

from experiments.grug.moe.model import GrugModelConfig as HfGrugModelConfig
from experiments.grug.moe.model import Transformer as HfTransformer
from experiments.grug.moe.model import grugmoe_inference_state_dict
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

# Files an export copies byte for byte from its base: save_pretrained re-serialises the tokenizer, and the base's
# serving and training templates must travel with the weights.
BASE_FILES = ("chat_template.jinja", "training_chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json")
GENERATION_CONFIG = {"bos_token_id": 128000, "eos_token_id": [128001, 128009], "pad_token_id": 128001}

_EMBEDDING_KEYS = (
    "model.embed_tokens.weight",
    "model.embed_norm.weight",
    "model.embed_gated_norm.down_proj.weight",
    "model.embed_gated_norm.up_proj.weight",
    "model.norm.weight",
    "model.final_gated_norm.down_proj.weight",
    "model.final_gated_norm.up_proj.weight",
    "lm_head.weight",
)
# (HF suffix, path in a Block, stored transposed in HF)
_BLOCK_TENSORS = (
    ("input_layernorm.weight", lambda b: b.rms_attn.weight, False),
    ("attn_gated_norm.down_proj.weight", lambda b: b.attn_gated_norm.w_down, True),
    ("attn_gated_norm.up_proj.weight", lambda b: b.attn_gated_norm.w_up, True),
    ("self_attn.q_proj.weight", lambda b: b.attn.w_q, True),
    ("self_attn.k_proj.weight", lambda b: b.attn.w_k, True),
    ("self_attn.v_proj.weight", lambda b: b.attn.w_v, True),
    ("self_attn.o_proj.weight", lambda b: b.attn.w_o, True),
    ("self_attn.attn_gate.weight", lambda b: b.attn.attn_gate, True),
    ("post_attention_layernorm.weight", lambda b: b.rms_mlp.weight, False),
    ("mlp_gated_norm.down_proj.weight", lambda b: b.mlp_gated_norm.w_down, True),
    ("mlp_gated_norm.up_proj.weight", lambda b: b.mlp_gated_norm.w_up, True),
    ("mlp.router.weight", lambda b: b.mlp.router, True),
    ("mlp.router.bias", lambda b: b.mlp.router_bias, False),
    ("mlp.experts.gate_proj.weight", lambda b: b.mlp.expert_mlp.w_gate, True),
    ("mlp.experts.up_proj.weight", lambda b: b.mlp.expert_mlp.w_up, True),
    ("mlp.experts.down_proj.weight", lambda b: b.mlp.expert_mlp.w_down, True),
)
_SHARED_EXPERT_TENSORS = (
    ("shared_expert.gate_proj.weight", lambda b: b.shared.w_gate, True),
    ("shared_expert.up_proj.weight", lambda b: b.shared.w_up, True),
    ("shared_expert.down_proj.weight", lambda b: b.shared.w_down, True),
)


def _block_tensors(config: GrugModelConfig) -> tuple:
    return _BLOCK_TENSORS + (_SHARED_EXPERT_TENSORS if config.shared_expert_intermediate_dim > 0 else ())


def _unstacked_blocks(model: Transformer) -> tuple[Any, ...]:
    if model.stacked_blocks is None:
        raise ValueError("HF conversion needs use_array_stacked_blocks=True.")

    def take_layer(value: Any, layer: int) -> Any:
        if isinstance(value, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(value.shape[1:], value.dtype)
        if isinstance(value, jax.Array):
            return value[layer]
        return value

    return tuple(
        jax.tree.map(lambda value, layer=layer: take_layer(value, layer), model.stacked_blocks.stacked)
        for layer in range(model.stacked_blocks.num_layers)
    )


def expected_hf_keys(config: GrugModelConfig) -> set[str]:
    keys = set(_EMBEDDING_KEYS)
    for layer in range(config.num_layers):
        keys.update(f"model.layers.{layer}.{suffix}" for suffix, _, _ in _block_tensors(config))
    return keys


def hf_state_dict(model: Transformer) -> dict[str, jax.Array]:
    """The HF tensor layout of a native model, with the router bias it currently holds."""
    source = cast(Any, model)
    unstacked = SimpleNamespace(
        token_embed=source.token_embed,
        embed_norm=source.embed_norm,
        embed_gated_norm=source.embed_gated_norm,
        output_proj=source.output_proj,
        blocks=_unstacked_blocks(model),
        final_norm=source.final_norm,
        final_gated_norm=source.final_gated_norm,
    )
    return grugmoe_inference_state_dict(cast(HfTransformer, unstacked))


def _checked(state_dict: dict[str, jax.Array], name: str, expected: Any, *, transposed: bool) -> jax.Array:
    value = state_dict[name]
    if transposed:
        value = jnp.swapaxes(value, -1, -2)
    if value.shape != expected.shape:
        raise ValueError(f"HF tensor {name!r} has shape {value.shape}; expected {expected.shape}.")
    return value


def native_from_hf_state_dict(template: Transformer, state_dict: dict[str, jax.Array]) -> tuple[Transformer, jax.Array]:
    """Load HF tensors into the native pytree; returns the model and ``pending_qb_betas = -router_bias`` (float32)."""
    expected = expected_hf_keys(template.config)
    missing = sorted(expected - set(state_dict))
    unexpected = sorted(set(state_dict) - expected)
    if missing or unexpected:
        raise ValueError(f"HF tensor schema mismatch: missing={missing}, unexpected={unexpected}")

    tensors = _block_tensors(template.config)
    blocks = []
    for layer, block in enumerate(_unstacked_blocks(template)):
        values = tuple(
            _checked(state_dict, f"model.layers.{layer}.{suffix}", select(block), transposed=transposed)
            for suffix, select, transposed in tensors
        )
        blocks.append(eqx.tree_at(lambda b: tuple(select(b) for _, select, _ in tensors), block, values))
    stacked = jax.tree.map(lambda *layers: jnp.stack(layers, axis=0), *blocks)

    model = eqx.tree_at(
        lambda m: (
            m.token_embed,
            m.embed_norm.weight,
            m.embed_gated_norm.w_down,
            m.embed_gated_norm.w_up,
            m.output_proj,
            m.stacked_blocks.stacked,
            m.final_norm.weight,
            m.final_gated_norm.w_down,
            m.final_gated_norm.w_up,
        ),
        template,
        (
            _checked(state_dict, "model.embed_tokens.weight", template.token_embed, transposed=False),
            _checked(state_dict, "model.embed_norm.weight", template.embed_norm.weight, transposed=False),
            _checked(
                state_dict, "model.embed_gated_norm.down_proj.weight", template.embed_gated_norm.w_down, transposed=True
            ),
            _checked(
                state_dict, "model.embed_gated_norm.up_proj.weight", template.embed_gated_norm.w_up, transposed=True
            ),
            _checked(state_dict, "lm_head.weight", template.output_proj, transposed=True),
            stacked,
            _checked(state_dict, "model.norm.weight", template.final_norm.weight, transposed=False),
            _checked(
                state_dict, "model.final_gated_norm.down_proj.weight", template.final_gated_norm.w_down, transposed=True
            ),
            _checked(
                state_dict, "model.final_gated_norm.up_proj.weight", template.final_gated_norm.w_up, transposed=True
            ),
        ),
    )
    pending_qb_betas = -jnp.stack(
        [state_dict[f"model.layers.{layer}.mlp.router.bias"].astype(jnp.float32) for layer in range(len(blocks))]
    )
    return model, pending_qb_betas


def with_router_bias_from_pending(model: Transformer, pending_qb_betas: jax.Array) -> Transformer:
    """The model with the router bias a training step would apply: ``-pending``, mean-centred per layer."""
    router_bias = -pending_qb_betas
    router_bias = router_bias - jnp.mean(router_bias, axis=-1, keepdims=True)
    return eqx.tree_at(lambda m: m.stacked_blocks.stacked.mlp.router_bias, model, router_bias)


def hf_model_config(config: GrugModelConfig) -> HfGrugModelConfig:
    """The HF-side config with the same values (the HF config class has a subset of the native fields)."""
    values = dataclasses.asdict(config)
    hf_fields = {field.name for field in dataclasses.fields(HfGrugModelConfig)}
    return draccus.decode(HfGrugModelConfig, {name: value for name, value in values.items() if name in hf_fields})


def load_hf_tensors(hf_dir: str, config: GrugModelConfig) -> dict[str, jax.Array]:
    converter = hf_model_config(config).hf_checkpoint_converter(ref_checkpoint=hf_dir)
    return converter.load_state_dict(hf_dir, dtype=jnp.bfloat16)


def load_native(checkpoint_path: str, config: GrugModelConfig, mesh: jax.sharding.Mesh) -> tuple[Transformer, jax.Array]:
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.PRNGKey(0))
    state = load_checkpoint(
        {
            "params": template,
            "pending_qb_betas": jax.ShapeDtypeStruct((config.num_layers, config.num_experts), jnp.float32),
        },
        checkpoint_path,
        mesh=mesh,
    )
    return state["params"], state["pending_qb_betas"]


def save_hf_export(model: Transformer, *, output_path: str, base_dir: str) -> None:
    """Write a vLLM-servable bf16 HF export with the base's config values, templates and tokenizer files."""
    model = jax.tree.map(lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value, model)
    source = cast(Any, model)
    hf_config = hf_model_config(model.config)
    export_model = HfTransformer(
        token_embed=source.token_embed,
        embed_norm=source.embed_norm,
        embed_gated_norm=source.embed_gated_norm,
        output_proj=source.output_proj,
        blocks=tuple(source.stacked_blocks.unstacked()),
        final_norm=source.final_norm,
        final_gated_norm=source.final_gated_norm,
        config=hf_config,
    )
    converter = (
        hf_config.hf_checkpoint_converter()
        .replaced(tokenizer=load_tokenizer(base_dir))
        .with_config_overrides({"dtype": "bfloat16"})
    )
    converter.save_pretrained(
        export_model,
        output_path,
        dtype=jnp.bfloat16,
        generation_config=GENERATION_CONFIG,
        chat_template=(Path(base_dir) / "chat_template.jinja").read_text(),
    )
    # every rank re-serialises the tokenizer files; copy the base's only after all of them are done
    multihost_utils.sync_global_devices("relay_sft_export_saved")
    if jax.process_index() == 0:
        for name in BASE_FILES:
            if (Path(base_dir) / name).is_file():
                shutil.copyfile(Path(base_dir) / name, Path(output_path) / name)
