# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Relay SFT of Grug 67B-A2B on a Slurm GPU cluster, one command per step.

    rows-to-parquet  rendered rows (jsonl) -> the one-shard Parquet
    prepare-cache    Parquet -> Levanter chat cache plus relay_cache.json (rows, tokens, packs, steps per epoch)
    import-hf        HF export of the base -> native step-0 init with pending_qb_betas = -router_bias
    train            the native Grug loop on every Slurm rank: frozen router bias, one cosine over every epoch
    export-hf        native checkpoint -> bf16 HF export with the base's config values, templates and tokenizer

``slurm/run_arm.sh`` chains them. Each rank of a multi-process step runs the same command; JAX discovers the Slurm
topology, and no Fray or Ray coordinator runs inside the allocation.
"""

import contextlib
import dataclasses
import json
import logging
from datetime import timedelta
from pathlib import Path

import click
import equinox as eqx
import jax
import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import set_mesh
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path, save_checkpoint
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from rigging.filesystem.storage_path import prefix_join

from experiments.june_tpu_67b_a2b.moe.model import Transformer
from experiments.june_tpu_67b_a2b.moe.train import GrugRunConfig, GrugTrainerConfig, run_grug_local
from experiments.post_training.relay_sft import recipe
from experiments.post_training.relay_sft.hf_conversion import (
    changed_leaves,
    leaf_abs_sums,
    load_hf_tensors,
    load_native,
    native_from_hf_state_dict,
    save_hf_export,
    with_router_bias_from_pending,
)
from experiments.post_training.relay_sft.prerendered import ROW_ADAPTER, PrerenderedChatFormat
from experiments.post_training.relay_sft.rows import (
    PackCount,
    count_packs,
    rows_table,
    write_rows_parquet,
)

logger = logging.getLogger(__name__)

CACHE_MANIFEST = "relay_cache.json"
COMPONENT = "relay"


@dataclasses.dataclass(frozen=True)
class CacheManifest:
    """What a relay cache was built from and how it packs; written by prepare-cache, checked by train."""

    parquet: str
    parquet_sha256: str
    tokenizer_sha256: str
    sequence_length: int
    batch_size: int
    row_adapter: str
    packs: PackCount

    @staticmethod
    def read(cache_path: str) -> "CacheManifest":
        record = json.loads((Path(cache_path) / CACHE_MANIFEST).read_text())
        return CacheManifest(**{**record, "packs": PackCount(**record["packs"])})


def relay_format() -> PrerenderedChatFormat:
    return PrerenderedChatFormat(chat_template=None, mask_user_turns=True, pack=None, max_tokens=recipe.SEQUENCE_LENGTH)


def relay_data_config(*, cache_path: str, tokenizer_path: str) -> LmDataConfig:
    fmt = relay_format()
    source = UrlDatasetSourceConfig(train_urls=[], cache_dir=cache_path, format=fmt, tags=[COMPONENT])
    return LmDataConfig(
        tokenizer=tokenizer_path,
        enforce_eos=True,
        auto_build_caches=False,
        components={
            COMPONENT: DatasetComponent(source=source, cache_dir=cache_path, format=fmt, tags=[COMPONENT], split="train")
        },
        train_weights={COMPONENT: 1.0},
        mixture_block_size=2048,
    )


def _cache_totals(cache_path: str) -> tuple[int, int]:
    stats = json.loads((Path(cache_path) / "train" / ".stats.json").read_text())
    return int(stats["total_elements"]), int(stats["total_tokens"])


def _check_cache(cache_path: str, tokenizer_path: str) -> CacheManifest:
    manifest = CacheManifest.read(cache_path)
    tokenizer_sha256 = recipe.sha256_file(Path(tokenizer_path) / "tokenizer.json")
    if manifest.tokenizer_sha256 != tokenizer_sha256:
        raise ValueError(f"{cache_path} was built with tokenizer {manifest.tokenizer_sha256}, not {tokenizer_sha256}.")
    if (manifest.sequence_length, manifest.batch_size) != (recipe.SEQUENCE_LENGTH, recipe.BATCH_SIZE):
        raise ValueError(
            f"{cache_path} was counted at {manifest.batch_size} x {manifest.sequence_length}, "
            f"the recipe trains {recipe.BATCH_SIZE} x {recipe.SEQUENCE_LENGTH}."
        )
    if manifest.row_adapter != ROW_ADAPTER:
        raise ValueError(f"{cache_path} holds {manifest.row_adapter} rows, expected {ROW_ADAPTER}.")
    rows, tokens = _cache_totals(cache_path)
    if (rows, tokens) != (manifest.packs.rows, manifest.packs.tokens):
        raise ValueError(
            f"{cache_path} holds {rows} rows / {tokens} tokens, but its Parquet had "
            f"{manifest.packs.rows} / {manifest.packs.tokens}: the cache is incomplete or from other rows."
        )
    return manifest


def _check_init(init_checkpoint_path: str, tokenizer_path: str) -> tuple[str, dict]:
    if not Path(init_checkpoint_path).is_absolute():
        raise ValueError(f"init path must be absolute: {init_checkpoint_path!r}")
    resolved = latest_checkpoint_path(init_checkpoint_path)
    step = json.loads((Path(resolved) / "metadata.json").read_text()).get("step")
    if step != 0:
        raise ValueError(f"{resolved} is step {step!r}; an init is the step-0 import of an HF export.")
    sidecar = recipe.read_base_sidecar(resolved)
    tokenizer_sha256 = recipe.sha256_file(Path(tokenizer_path) / "tokenizer.json")
    if sidecar["tokenizer_sha256"] != tokenizer_sha256:
        raise ValueError(f"{resolved} was imported from a base with another tokenizer than {tokenizer_path}.")
    return resolved, sidecar


def _check_output(output_path: str, *, inputs: tuple[str, ...], resume: bool) -> None:
    path = Path(output_path)
    if not path.is_absolute():
        raise ValueError(f"output path must be absolute: {output_path!r}")
    resolved = path.resolve()
    for input_path in inputs:
        other = Path(input_path).resolve()
        if resolved == other or resolved in other.parents or other in resolved.parents:
            raise ValueError(f"output {resolved} overlaps input {other}")
    if resume:
        latest_checkpoint_path(str(path / "checkpoints"), str(path / "checkpoints-tmp"))
    elif path.exists():
        raise FileExistsError(f"{path} exists; pass --resume to continue the run it holds.")


def relay_run_config(
    *,
    init_checkpoint_path: str,
    cache_path: str,
    tokenizer_path: str,
    output_path: str,
    run_id: str,
    epochs: int,
    learning_rate: float,
    steps: int | None,
    resume: bool,
) -> GrugRunConfig:
    """Check every input, then build the run: ``epochs`` passes over the cache's packs on one cosine.

    ``steps`` trains only the first steps of that schedule (a smoke test, or a run continued later with --resume);
    one permanent checkpoint is kept per epoch.
    """
    resolved_init, sidecar = _check_init(init_checkpoint_path, tokenizer_path)
    manifest = _check_cache(cache_path, tokenizer_path)
    _check_output(output_path, inputs=(resolved_init, cache_path, tokenizer_path), resume=resume)
    epoch_steps = manifest.packs.epoch_steps
    schedule_steps = epochs * epoch_steps
    num_train_steps = schedule_steps if steps is None else steps
    if not 0 < num_train_steps <= schedule_steps:
        raise ValueError(f"steps must lie in 1..{schedule_steps} ({epochs} epochs of {epoch_steps}), got {steps}")
    warmup_steps = recipe.warmup_steps_for(schedule_steps)
    logger.info(
        "relay SFT: %d steps of a %d-step schedule (%d epochs x %d), warmup %d, lr %g, base %s",
        num_train_steps,
        schedule_steps,
        epochs,
        epoch_steps,
        warmup_steps,
        learning_rate,
        sidecar["hf_base"],
    )
    trainer = TrainerConfig(
        id=run_id,
        seed=recipe.SEED,
        train_batch_size=recipe.BATCH_SIZE,
        per_device_parallelism=-1,
        num_train_steps=num_train_steps,
        mp=jmp.get_policy(recipe.MIXED_PRECISION),
        tracker=WandbConfig(
            project="relay_sft",
            name=run_id,
            group="grug-67b-a2b-relay-sft",
            tags=["relay_sft", "67b_a2b"],
            mode="offline",
        ),
        use_explicit_mesh_axes=True,
        # every Slurm rank is a one-device JAX slice: the ranks form replica_dcn, Grug builds its own compute mesh
        mesh=MeshConfig(axes={"expert": 1}, compute_mapping={"batch": ["replica_dcn", "data", "expert"]}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=prefix_join(output_path, "checkpoints"),
            temporary_base_path=prefix_join(output_path, "checkpoints-tmp"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=30),
            keep=[{"every": epoch_steps}],
        ),
        load_checkpoint=None,
        load_checkpoint_path=None,
        initialize_from=resolved_init,
    )
    return GrugRunConfig(
        model=recipe.model_config_for_base(sidecar["config"], max_seq_len=recipe.SEQUENCE_LENGTH),
        data=relay_data_config(cache_path=cache_path, tokenizer_path=tokenizer_path),
        # unused by the local loop; GrugRunConfig requires it for Fray dispatch
        resources=ResourceConfig.with_gpu("GH200", count=1, replicas=recipe.DEVICES),
        optimizer=recipe.relay_optimizer(learning_rate=learning_rate, warmup_steps=warmup_steps),
        trainer=GrugTrainerConfig(
            trainer=trainer,
            z_loss_weight=recipe.Z_LOSS_WEIGHT,
            ema_beta=None,
            log_every=1,
            replica_axis_size=1,
            model_axis_size=1,
            expert_axis_size=recipe.EXPERT_AXIS_SIZE,
            sft_weights_only_init=True,
            freeze_router_bias=True,
            schedule_steps=schedule_steps,
        ),
        eval=None,
    )


def _native_mesh():
    return compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=1)


def _devices(distributed: bool):
    """Initialise JAX over the Slurm ranks, or keep a single process on its CPU device."""
    if distributed:
        DistributedConfig().initialize()
        return contextlib.nullcontext()
    return use_cpu_device()


@click.group()
def main() -> None:
    logging.basicConfig(level=logging.INFO)


@main.command("rows-to-parquet")
@click.argument("rows", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
def rows_to_parquet_command(rows: Path, out_dir: Path) -> None:
    """Check rendered rows and write OUT_DIR/train-00000-of-00001.parquet."""
    table = rows_table(rows, max_tokens=recipe.SEQUENCE_LENGTH, bos_id=recipe.BOS_ID, vocab_size=recipe.VOCAB_SIZE)
    shard = write_rows_parquet(table, out_dir)
    packs = count_packs(shard, sequence_length=recipe.SEQUENCE_LENGTH, batch_size=recipe.BATCH_SIZE)
    click.echo(json.dumps({"parquet": str(shard), **dataclasses.asdict(packs)}))


@main.command("prepare-cache")
@click.option("--parquet", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tokenizer", required=True, type=click.Path(exists=True, file_okay=False), help="The base's HF dir.")
@click.option("--cache-path", required=True)
def prepare_cache_command(parquet: str, tokenizer: str, cache_path: str) -> None:
    """Build the chat cache from one Parquet shard and record how it packs."""
    packs = count_packs(Path(parquet), sequence_length=recipe.SEQUENCE_LENGTH, batch_size=recipe.BATCH_SIZE)
    tokenize(
        TokenizeConfig(
            train_paths=[str(Path(parquet).resolve())],
            validation_paths=[],
            cache_path=cache_path,
            tokenizer=tokenizer,
            tags=[COMPONENT],
            format=relay_format(),
            max_workers=1,
            worker_resources=ResourceConfig(cpu=16, ram="64g", disk="20g"),
        )
    )
    manifest = CacheManifest(
        parquet=str(Path(parquet).resolve()),
        parquet_sha256=recipe.sha256_file(Path(parquet)),
        tokenizer_sha256=recipe.sha256_file(Path(tokenizer) / "tokenizer.json"),
        sequence_length=recipe.SEQUENCE_LENGTH,
        batch_size=recipe.BATCH_SIZE,
        row_adapter=ROW_ADAPTER,
        packs=packs,
    )
    (Path(cache_path) / CACHE_MANIFEST).write_text(json.dumps(dataclasses.asdict(manifest), indent=2))
    _check_cache(cache_path, tokenizer)
    click.echo(json.dumps(dataclasses.asdict(manifest)))
    click.echo("RELAY_CACHE_OK")


@main.command("import-hf")
@click.option("--hf-checkpoint", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output-path", required=True, help="New native checkpoint directory, conventionally .../step-0.")
@click.option("--distributed/--single-process", default=True, help="Shard the conversion over the Slurm ranks.")
def import_hf_command(hf_checkpoint: str, output_path: str, distributed: bool) -> None:
    """Import the base's HF export as a step-0 native init with its snowball_base.json sidecar."""
    if Path(output_path).exists():
        raise click.ClickException(f"{output_path} exists")
    sidecar = recipe.base_sidecar(hf_checkpoint)
    config = recipe.model_config_for_base(sidecar["config"])
    with _devices(distributed):
        mesh = _native_mesh()
        with set_mesh(mesh):
            template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.PRNGKey(0))
            params, pending_qb_betas = native_from_hf_state_dict(template, load_hf_tensors(hf_checkpoint, config))
            manager = GlobalAsyncCheckpointManager()
            save_checkpoint(
                {"params": params, "pending_qb_betas": pending_qb_betas},
                step=0,
                checkpoint_path=output_path,
                manager=manager,
                is_temporary=False,
            )
            manager.wait_until_finished()
            if jax.process_index() == 0:
                (Path(output_path) / recipe.BASE_SIDECAR).write_text(json.dumps(sidecar, indent=2, sort_keys=True))
            reloaded, reloaded_pending = load_native(output_path, config, mesh)
            parameter_count = sum(leaf.size for leaf in jax.tree.leaves(reloaded) if isinstance(leaf, jax.Array))
            if parameter_count != recipe.NATIVE_PARAMETERS:
                raise ValueError(f"reloaded {parameter_count} parameters, expected {recipe.NATIVE_PARAMETERS}")
            if not bool(jax.device_get((reloaded_pending == pending_qb_betas).all())):
                raise ValueError("reloaded pending_qb_betas differ from the base's router bias")
            # A write can report success and store nothing (see tensorstore_serialization), so compare every array.
            changed = changed_leaves(leaf_abs_sums(params), leaf_abs_sums(reloaded))
            if changed:
                raise ValueError(f"the saved init differs from the converted model in {changed}")
    click.echo(f"RELAY_IMPORT_OK {output_path} qk_mult={sidecar['qk_mult']} parameters={parameter_count}")


@main.command("train")
@click.option("--init-checkpoint-path", required=True)
@click.option("--cache-path", required=True)
@click.option("--tokenizer", required=True, help="The base's HF dir (the tokenizer the rows were rendered with).")
@click.option("--output-path", required=True)
@click.option("--run-id", required=True)
@click.option("--epochs", type=click.IntRange(min=1), default=recipe.EPOCHS, show_default=True)
@click.option("--learning-rate", type=float, default=recipe.LEARNING_RATE, show_default=True)
@click.option("--steps", type=click.IntRange(min=1), default=None, help="Train only the first STEPS of the schedule.")
@click.option("--resume", is_flag=True, help="Continue the run under --output-path from its latest checkpoint.")
@click.option("--check-only", is_flag=True, help="Check every input and print the plan without training.")
def train_command(
    init_checkpoint_path: str,
    cache_path: str,
    tokenizer: str,
    output_path: str,
    run_id: str,
    epochs: int,
    learning_rate: float,
    steps: int | None,
    resume: bool,
    check_only: bool,
) -> None:
    """Train on the cache from the init; every Slurm rank runs this once."""
    config = relay_run_config(
        init_checkpoint_path=init_checkpoint_path,
        cache_path=cache_path,
        tokenizer_path=tokenizer,
        output_path=output_path,
        run_id=run_id,
        epochs=epochs,
        learning_rate=learning_rate,
        steps=steps,
        resume=resume,
    )
    if check_only:
        click.echo(
            f"steps={config.trainer.trainer.num_train_steps} schedule_steps={config.trainer.schedule_steps} "
            f"warmup={config.optimizer.warmup} keep_every={config.trainer.trainer.checkpointer.keep}"
        )
        click.echo("RELAY_TRAIN_CHECK_OK")
        return
    run_grug_local(config)


@main.command("export-hf")
@click.option("--checkpoint", required=True, help="A native step-N directory.")
@click.option("--base", required=True, type=click.Path(exists=True, file_okay=False), help="The base's HF dir.")
@click.option("--output-path", required=True)
@click.option("--distributed/--single-process", default=True, help="Shard the conversion over the Slurm ranks.")
def export_hf_command(checkpoint: str, base: str, output_path: str, distributed: bool) -> None:
    """Export a native checkpoint as a bf16 HF model with the base's config values, templates and tokenizer."""
    if Path(output_path).exists():
        raise click.ClickException(f"{output_path} exists")
    config = recipe.model_config_for_base(recipe.read_base_hf_config(base))
    with _devices(distributed):
        mesh = _native_mesh()
        with set_mesh(mesh):
            params, pending_qb_betas = load_native(checkpoint, config, mesh)
            model = with_router_bias_from_pending(params, pending_qb_betas)
            save_hf_export(model, output_path=output_path, base_dir=base)
    click.echo(f"RELAY_EXPORT_OK {output_path}")


if __name__ == "__main__":
    main()
