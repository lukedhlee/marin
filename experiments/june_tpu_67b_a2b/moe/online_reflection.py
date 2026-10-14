"""One fresh reflection batch, one accumulated AdamW step, full-state checkpoint.

Separate entry point: existing SFT training is unchanged. Execute under the existing
Horizon GPU environment and srun; this program never allocates cluster resources.
"""

# ruff: noqa: PLC0415 -- defer accelerator imports for offline schema/math validation
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import shutil
from pathlib import Path


def training_fingerprint(training: dict, base_config: dict, mesh_config: dict) -> str:
    payload = {"training": training, "base_config": base_config, "mesh": mesh_config}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def weighted_add(accumulator, gradient, fraction):
    import jax

    return jax.tree.map(lambda acc, grad: acc + grad * fraction, accumulator, gradient)


def apply_optimizer_update(state, params, grads, optimizer, scale):
    import jax
    import optax

    grads = jax.tree.map(lambda grad: grad * scale, grads)
    updates, next_opt = optimizer.update(grads, state.opt_state, params)
    return dataclasses.replace(
        state, params=optax.apply_updates(params, updates), opt_state=next_opt, step=state.step + 1
    )


def run(request: dict, response_path: str) -> None:
    # Imports kept inside run so schema/fingerprint checks need no accelerator stack.
    import draccus
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jmp
    import numpy as np
    import optax
    from haliax.partitioning import set_mesh
    from jax.experimental import multihost_utils
    from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from levanter.checkpoint import load_checkpoint, save_checkpoint
    from levanter.data.text.examples import GrugLmExample
    from levanter.distributed import DistributedConfig
    from levanter.grug.attention import AttentionMask
    from levanter.grug.sharding import _batch_axes, compact_grug_mesh
    from levanter.tokenizers import load_tokenizer

    from experiments.grug.moe.model import GrugModelConfig as ExportConfig
    from experiments.grug.moe.model import Transformer as ExportTransformer
    from experiments.june_tpu_67b_a2b.moe.snowball_chat_recipe import (
        read_base_hf_config,
        read_init_base_sidecar,
        snowball_model_config_for_base,
    )
    from experiments.june_tpu_67b_a2b.moe.train import (
        _apply_qb_betas,
        apply_gpu_runtime_defaults,
        init_weights_only_from_checkpoint,
        initial_state,
    )

    training, backend = request["config"]["training"], request["config"]["backend"]
    iteration = request["iteration"]
    if type(iteration) is not int or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    if training["loss_reduction"] != "global_token_mean":
        raise ValueError("global_token_mean is required")
    if iteration > training["schedule_steps"]:
        raise ValueError("iteration exceeds pinned schedule horizon")
    if bool(request.get("optimizer_before")) != (iteration > 1):
        raise ValueError("missing/extraneous optimizer_before")
    rows_bytes = Path(request["rows"]).read_bytes()
    digest = hashlib.sha256(rows_bytes).hexdigest()
    if digest != request["source_digest"]:
        raise ValueError("rows changed after adapter validation")
    rows = [json.loads(line) for line in rows_bytes.splitlines() if line.strip()]
    length, size = training["sequence_length"], training["microbatch_size"]
    if any(len(row["ids"]) != len(row["loss"]) or len(row["ids"]) > length for row in rows):
        raise ValueError("malformed/overlength rows; never truncate")
    if not rows or len(rows) != request["row_count"]:
        raise ValueError("empty batch or row census mismatch")
    if len({row["sid"] for row in rows}) != len(rows):
        raise ValueError("duplicate reflection identities")
    if any(
        row["loss"][0] != 0 or not sum(row["loss"]) or any(value not in (0, 1) for value in row["loss"]) for row in rows
    ):
        raise ValueError("invalid next-token mask")
    total_tokens = sum(sum(row["loss"]) for row in rows)
    if total_tokens != request["supervised_tokens"]:
        raise ValueError("supervised token census differs")
    base = backend["base_model"]
    base_config = read_base_hf_config(base)
    model_config = dataclasses.replace(
        snowball_model_config_for_base(base_config, max_seq_len=length), router_z_loss_coef=0.0
    )
    mesh_cfg = backend.get("mesh", {"expert_axis_size": 4, "replica_axis_size": 1, "model_axis_size": 1})
    fingerprint = training_fingerprint(training, base_config, mesh_cfg)
    previous = None
    if iteration > 1:
        previous = json.loads((Path(request["optimizer_before"]) / "online_state.json").read_text())
        if previous["fingerprint"] != fingerprint or previous["iteration"] != iteration - 1:
            raise ValueError("optimizer configuration or iteration changed")
        if Path(previous["model_after"]).resolve() != Path(request["model_before"]).resolve():
            raise ValueError("rollout model does not match previous optimizer weights")
        if digest in previous["source_digests"]:
            raise ValueError("fresh batch required; source digest was already trained")
    else:
        if Path(request["model_before"]).resolve() != Path(base).resolve():
            raise ValueError("first rollout model must equal pinned base")
        sidecar = read_init_base_sidecar(backend["native_init"])
        if sidecar is None or sidecar["config"] != base_config:
            raise ValueError("native initialization lacks matching base configuration")
        if Path(sidecar["hf_base"]).resolve() != Path(base).resolve():
            raise ValueError("native initialization names a different HF base")
        for filename, field in (("config.json", "config_sha256"), ("tokenizer.json", "tokenizer_sha256")):
            if hashlib.sha256((Path(base) / filename).read_bytes()).hexdigest() != sidecar[field]:
                raise ValueError(f"native initialization {filename} provenance mismatch")
        if not sidecar["pending_qb_betas_from_router_bias"]:
            raise ValueError("native initialization did not preserve the base router biases")

    apply_gpu_runtime_defaults()
    DistributedConfig().initialize()
    mp = jmp.get_policy("p=f32,c=bfloat16")
    transformations = []
    if training["grad_clip"] is not None:
        transformations.append(optax.clip_by_global_norm(training["grad_clip"]))
    # Constant learning rate; Adam's bias-correction counter remains in the native state.
    transformations.append(
        optax.adamw(
            training["lr"],
            b1=training["beta1"],
            b2=training["beta2"],
            eps=training["epsilon"],
            weight_decay=training["weight_decay"],
        )
    )
    optimizer = optax.chain(*transformations)
    mesh = compact_grug_mesh(**mesh_cfg)
    axes = _batch_axes(mesh)
    shards = int(np.prod([mesh.shape[axis] for axis in axes]))
    if size <= 0 or size % shards:
        raise ValueError(f"global microbatch_size must be divisible by batch mesh size {shards}")
    output = Path(request["output_dir"])
    native, export = output / "native", output / "hf"
    if native.exists() or export.exists():
        raise ValueError("refusing to overwrite an update")
    with set_mesh(mesh):
        state = jax.jit(
            lambda: initial_state(model_config, optimizer=optimizer, mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)
        )()
        if previous is None:
            state = init_weights_only_from_checkpoint(state, backend["native_init"], mesh=mesh, load_ema=False)
        else:
            # Strict full-state restore: params, Adam moments/counters, step and router state.
            state = load_checkpoint(state, request["optimizer_before"], mesh=mesh, allow_partial=False)
        before = int(state.step)
        if before != iteration - 1:
            raise ValueError("checkpoint step does not match requested iteration")
        params = _apply_qb_betas(state.params, state.pending_qb_betas)

        @jax.jit
        def loss_and_grad(params, batch):
            def loss_fn(p):
                return mp.cast_to_compute(p).next_token_loss(
                    batch.tokens, batch.loss_weight, mask=batch.attn_mask, reduction="mean"
                )

            return jax.value_and_grad(loss_fn)(params)

        add_gradient = jax.jit(weighted_add)

        accumulated = jax.tree.map(jnp.zeros_like, params)
        loss_value = 0.0
        for offset in range(0, len(rows), size):
            subset = rows[offset : offset + size]
            tokens = np.full((size, length), 128001, dtype=np.int32)
            weights = np.zeros((size, length), dtype=np.float32)
            batch_tokens = 0
            for index, row in enumerate(subset):
                n = len(row["ids"])
                tokens[index, :n] = row["ids"]
                # Input masks label target-token positions; Grug masks label predictions.
                weights[index, : n - 1] = row["loss"][1:]
                batch_tokens += sum(row["loss"])
            sharding = NamedSharding(mesh, P(axes, None))
            batch = GrugLmExample(
                jax.make_array_from_callback(tokens.shape, sharding, lambda index, values=tokens: values[index]),
                jax.make_array_from_callback(weights.shape, sharding, lambda index, values=weights: values[index]),
                AttentionMask.causal(),
            )
            loss, grads = loss_and_grad(params, batch)
            fraction = batch_tokens / total_tokens
            accumulated = add_gradient(accumulated, grads, fraction)
            loss_value += float(loss) * fraction
            del grads, batch
        if not np.isfinite(loss_value):
            raise ValueError("nonfinite loss; update was not applied")

        @jax.jit
        def apply_update(state, params, grads):
            return apply_optimizer_update(state, params, grads, optimizer, training["gradient_scale"])

        state = apply_update(state, params, accumulated)
        jax.block_until_ready(state)
        if int(state.step) != iteration:
            raise ValueError("expected exactly one optimizer step")
        # Synchronous save: metadata only appears after all tensors are written. The default commit barrier (300 s)
        # fails when ranks finish their shards minutes apart on a congested /scratch (2026-10-13: BarrierTimeoutError
        # after ~1 h at ~80 MB/s), so the barrier waits up to six hours.
        manager = GlobalAsyncCheckpointManager(timeout_secs=6 * 3600)
        save_checkpoint(state, iteration, str(native), manager=manager, is_temporary=False)
        manager.wait_until_finished()
        del accumulated, params
        export_params = _apply_qb_betas(state.params, state.pending_qb_betas)
        del state
        gc.collect()
        export_params = jax.tree.map(
            lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value, export_params
        )
        export_config = snowball_model_config_for_base(base_config)
        fields = {field.name for field in dataclasses.fields(ExportConfig)}
        export_config = draccus.decode(
            ExportConfig, {key: value for key, value in dataclasses.asdict(export_config).items() if key in fields}
        )
        source = export_params
        export_model = ExportTransformer(
            token_embed=source.token_embed,
            embed_norm=source.embed_norm,
            embed_gated_norm=source.embed_gated_norm,
            output_proj=source.output_proj,
            blocks=tuple(source.stacked_blocks.unstacked()),
            final_norm=source.final_norm,
            final_gated_norm=source.final_gated_norm,
            config=export_config,
        )
        converter = export_config.hf_checkpoint_converter().replaced(tokenizer=load_tokenizer(base))
        converter.with_config_overrides({"dtype": "bfloat16"}).save_pretrained(
            export_model,
            str(export),
            dtype=jnp.bfloat16,
            generation_config={"bos_token_id": 128000, "eos_token_id": [128001, 128009], "pad_token_id": 128001},
            chat_template=(Path(base) / "chat_template.jinja").read_text(),
            save_tokenizer=jax.process_index() == 0,
        )
        multihost_utils.sync_global_devices("online_export_saved")
        if jax.process_index() == 0:
            for name in (
                "training_chat_template.jinja",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ):
                if (Path(base) / name).is_file():
                    shutil.copyfile(Path(base) / name, export / name)
            provenance = {
                "iteration": iteration,
                "fingerprint": fingerprint,
                "model_after": str(export),
                "source_digests": (previous["source_digests"] if previous else []) + [digest],
            }
            (native / "online_state.json").write_text(json.dumps(provenance, indent=2) + "\n")
            result = {
                "model_after": str(export),
                "optimizer_after": str(native),
                "optimizer_steps_before": before,
                "optimizer_steps_after": iteration,
                "weights_version": iteration,
                "supervised_tokens": total_tokens,
                "loss": loss_value,
                "source_digest": digest,
            }
            Path(response_path).write_text(json.dumps(result, indent=2) + "\n")
        multihost_utils.sync_global_devices("online_response_written")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.request).read_text()), args.response)


if __name__ == "__main__":
    main()
