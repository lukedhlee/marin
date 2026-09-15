# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""R2E-Gym teacher-trace SFT of the pre-RL Snowball checkpoint (Stage-3 Nemotron-Terminal, step 1888).

One stage on the general ``experiments.sft`` launcher, composed exactly like ``sft_67b_a2b_2stage``:
a ``GrugModel`` source (native weights-only init from the Stage-3 checkpoint, fresh optimizer) plus one
``DatasetSpec`` of Terminus-2 teacher traces on R2E-Gym tasks. Everything else is Ben's Stage-3
control recipe (marin #8225; the launcher restore is PR #8172):

  * chat template = the Marin template, so the Terminus-2 JSON actions and the literal ``<think>``
    tags stay verbatim in the assistant span (this is the surface the Harbor RL harness serves and
    parses), loss on assistant spans only, sequence packing on;
  * seq 32,768, global batch 64 on 8x H100x8 (cw-us-east-02a), the ``_model`` / mesh geometry of the
    2-stage file;
  * AdamH 5e-6 on every parameter group, cosine, 3% warmup, min-lr ratio 0.1, beta 0.9/0.95, wd 0;
  * one packed epoch (``sft_step`` derives the step count from the chat cache, marin #7244).

The dataset is a launch flag so the same recipe runs on any Terminus-2 trace repo whose rows carry a
``conversations`` list of ``{role, content}`` turns (the OT-Agent datagen schema). The default is the
largest public GLM-4.7 R2E-Gym trace repo; it is UNFILTERED against the R2E-Gym v2 validation splits
(idval / oodval / heldout), so use it for smokes only and point ``--dataset-id`` at the split-filtered
set for the real run.

Submit (cw-us-east-02a, preemptible; MARIN_PREFIX must be s3://marin-us-east-02a/marin; AWS creds are
injected in-pod via the iris-task-env secret -- do not forward AWS_*)::

    cd ~/marin && source secrets.env
    export KUBECONFIG=~/.kube/coreweave-iris-gpu
    uv run iris --cluster=cw-us-east-02a job run --job-name snowball-r2egym-sft-coord \\
      --cpu 1 --memory 2G --extra cpu --priority interactive --max-retries 10 --no-wait \\
      -e MARIN_PREFIX s3://marin-us-east-02a/marin -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_r2egym --stage smoke --version dev --run

Drop ``--run`` to print the lowered plan without launching; ``--version`` is required.
"""

import dataclasses

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage import (
    _BATCH,
    _NODES,
    _SEQ,
    _WANDB_PROJECT,
    _gpu_resources,
    _grug_source,
    _optimizer,
)
from experiments.marin_tokenizer import MARIN_CHAT_TEMPLATE
from experiments.sft.launcher import DatasetSpec, SFTSpec, sft_step

# --- Init: the pre-RL Snowball checkpoint = Stage-3 Nemotron-Terminal SFT (1,888 steps), native
# Levanter Grug checkpoint on the CoreWeave mirror. ``GrugModel`` resolves the latest ``step-N`` under it.
# HF export of the same weights: laion/snowball-67b-a2b-sft-s3-nemotron-terminal-step1888 @ 680bec3a.
_SNOWBALL_S3_CKPT: str = (
    "s3://marin-us-east-02a/marin/grug/snowball_step105149_sft_s3_nemotron_terminal_steps1888/2026.08.14.1/checkpoints/"
)

# --- Optimizer: the Stage-3 agentic setting (PR #8172 ``_agentic_optimizer``) -- the 2-stage AdamH config
# with both groups at 5e-6 instead of 5e-5.
_AGENTIC_LR: float = 5e-6
_agentic_optimizer = dataclasses.replace(_optimizer, learning_rate=_AGENTIC_LR, adam_lr=_AGENTIC_LR)

# --- Dataset: Terminus-2 teacher traces on R2E-Gym in the OT-Agent datagen schema. The default is the
# raw GLM-4.7 pool (8,578 rows, HEAD 2026-02-23) and is NOT filtered against the v2 validation splits.
_DEFAULT_DATASET_ID: str = "DCAgent/exp-syh-r2egym-askllm-constrained_glm_4.7_traces_jupiter_cleaned"
_DEFAULT_DATASET_REVISION: str = "d13cd4d"
_DEFAULT_DATASET_SLUG: str = "r2egym_glm47_askllm_raw"

_RUN_ID: str = "snowball_s3_step1888_sft_r2egym_teacher"
_SMOKE_RUN_ID: str = "snowball_s3_step1888_sft_r2egym_smoke"
_SMOKE_STEPS: int = 8  # clear the first jit_train_step at the target geometry and bank a checkpoint


def _dataset(dataset_id: str, revision: str, slug: str) -> DatasetSpec:
    return DatasetSpec(
        slug=slug,
        hf_dataset_id=dataset_id,
        revision=revision,
        adapter_kwargs=dict(conversation_column="conversations"),  # role/content, user/assistant defaults
        weight=1.0,
    )


def _spec(
    *,
    name: str,
    version: str,
    dataset: DatasetSpec,
    stage: str,
    steps: int | None = None,
    epochs: int | None = None,
    save_interval_minutes: int = 60,
    checkpoint_keep: list[dict] | None = None,
) -> SFTSpec:
    return SFTSpec(
        name=name,
        version=version,
        model=_grug_source(
            _SNOWBALL_S3_CKPT,
            stage=stage,
            seq=_SEQ,
            save_interval_minutes=save_interval_minutes,
            checkpoint_keep=checkpoint_keep,
        ),
        chat_template=MARIN_CHAT_TEMPLATE,
        datasets=[dataset],
        optimizer=_agentic_optimizer,
        seq_len=_SEQ,
        batch_size=_BATCH,
        num_train_steps=steps,
        num_train_epochs=epochs,
        wandb_project=_WANDB_PROJECT,
    )


def build_full(dataset: DatasetSpec, version: str | None = None, *, epochs: int = 1) -> ArtifactStep[LevanterCheckpoint]:
    """The real run: ``epochs`` packed epochs over ``dataset``, weights-only init from Stage 3."""
    step_name = f"grug/{_RUN_ID}"
    version = resolve_version(step_name, version)
    spec = _spec(
        name=user_namespaced_name(step_name, version),
        version=version,
        dataset=dataset,
        stage="r2egym_teacher",
        epochs=epochs,
        checkpoint_keep=[{"every": 100}],
    )
    return sft_step(spec, _gpu_resources(_NODES))


def build_smoke(dataset: DatasetSpec, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Smoke at the full geometry: native S3 load -> chat cache -> first jit_train_step -> a mid-run save."""
    step_name = f"grug/{_SMOKE_RUN_ID}"
    version = resolve_version(step_name, version)
    spec = _spec(
        name=user_namespaced_name(step_name, version),
        version=version,
        dataset=dataset,
        stage="r2egym_smoke",
        steps=_SMOKE_STEPS,
        save_interval_minutes=5,
        checkpoint_keep=[{"every": 4}],
    )
    return sft_step(spec, _gpu_resources(_NODES))


@click.command()
@click.option(
    "--stage",
    type=click.Choice(["smoke", "full"]),
    default="smoke",
    show_default=True,
    help="smoke = a few steps at the full geometry; full = the epoch run.",
)
@click.option("--dataset-id", default=_DEFAULT_DATASET_ID, show_default=True, help="HF trace repo (OT-Agent schema).")
@click.option("--dataset-revision", default=_DEFAULT_DATASET_REVISION, show_default=True, help="7-char HF commit pin.")
@click.option("--dataset-slug", default=_DEFAULT_DATASET_SLUG, show_default=True, help="Mixture key / cache name.")
@click.option("--epochs", default=1, show_default=True, type=int, help="Packed epochs for --stage full.")
@build_options
def main(
    stage: str, dataset_id: str, dataset_revision: str, dataset_slug: str, epochs: int
) -> ArtifactStep[LevanterCheckpoint]:
    dataset = _dataset(dataset_id, dataset_revision, dataset_slug)
    if stage == "smoke":
        return build_smoke(dataset)
    return build_full(dataset, epochs=epochs)


if __name__ == "__main__":
    main()
