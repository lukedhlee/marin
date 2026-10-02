# Relay SFT for Grug 67B-A2B

Supervised fine-tuning of Grug 67B-A2B 09-21 (`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`) on *relay*
traces: the student model works a terminal task until it hands over, and Qwen3.8-27B finishes it. Only the teacher's
turns are trained. On the same task slots, relay traces beat traces where Qwen3.8 plays every turn by 6 points on
Terminal-Bench 2.1, 22 on SWE-bench Verified (random 100) and 10 on OpenThoughts-TBLite, and they are the only data
found to move Terminal-Bench 2.1.

| model | TB2.1 | SWE-bench Verified (100) | TBLite |
|---|---|---|---|
| 09-21 base | 6.5 | 13.8 | 11.7 |
| `laion/snowball-67b-a2b-relay-sft-allkimi-step1203` (H8: 09-21 + `h8_allkimi`) | 18.8 | 46.6 | 27.2 |
| `laion/snowball-67b-a2b-relay-sft-acont-step999` (H9: arm A + `h9_acont`) | 20.8 | 45.3 | 28.2 |

pass@1 in percent, mean of 3 runs under Terminus-2 with a 65,536-token context. The rows, already tokenized, are in
`laion/snowball-relay-sft-rows`; the readable traces, labels and the rows left out are in
`laion/calibforge-relay-traces`. Traces are generated and rendered by OpenThoughts-Agent (`data/relay/`, see its
`HANDOFF.md`).

## Recipe

- Rows arrive as `ids` plus a 0/1 `loss` per token (`prerendered.py`); the cache copies them, and the chat dataset's
  packing and mask shift apply unchanged.
- The base is imported with `pending_qb_betas = -router_bias` and the router bias stays frozen there
  (`freeze_router_bias`): 09-21 kept its bias frozen through its own fine-tuning, and re-deriving it from narrow SFT
  batches pulls the expert balance toward the SFT data.
- AdamH, LR 3e-4 for both parameter groups, 5 % linear warmup, one cosine over 3 passes over the packs, z-loss 1e-4,
  seed 0. 16 sequences of 65,536 tokens per step on 16 GPUs (4 nodes, one JAX device per Slurm rank, experts
  sharded 8 ways): 7.3 s per step on Horizon's GB200.
- The importer records the base's `config.json` (`qk_mult` 1.75, `max_position_embeddings` 262,144 for 09-21) in a
  `snowball_base.json` sidecar inside the init; training and export build the model config from it.

## Running on Jupiter

```bash
cd <marin checkout>/experiments/post_training/relay_sft/slurm
export RELAY_CLUSTER=jupiter RELAY_ACCOUNT=<project> RELAY_ROOT=/e/data1/<group>/$USER/relay-sft

bash build_env.sh                                    # login node, once: uv sync, NCCL 2.30.7, import smoke

hf download open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 --local-dir $RELAY_ROOT/models/grug-0921
hf download laion/snowball-relay-sft-rows --repo-type dataset --local-dir $RELAY_ROOT/data/relay-sft-rows

BASE=$RELAY_ROOT/models/grug-0921 PARQUET=$RELAY_ROOT/data/relay-sft-rows/h8_allkimi/train-00000-of-00001.parquet \
NAME=h8-repro bash run_arm.sh                        # inside tmux: prep + import -> train -> exports
```

`run_arm.sh` submits `prep.sbatch` (cache, 1 node) and `import.sbatch` (09-21 to a native init, 4 nodes, once per
base) side by side, then `train.sbatch` (4 nodes, 1,203 steps for `h8_allkimi`, about 2.5 h), then one
`export.sbatch` per kept checkpoint (one per pass). Exports land in `$RELAY_ROOT/runs/<NAME>/export-step-*-hf-bf16`
and serve with the Grug vLLM fork. Rerunning resumes: finished steps are skipped and a training job continues from
its latest checkpoint.

- New rows: `ROWS=<rendered jsonl>` builds `PARQUET` first (`relay_sft.py rows-to-parquet`).
- Continuing from a fine-tuned model (as H9 continued arm A): set `BASE` to its HF export.
- `STEPS=30` trains the first 30 steps of the full schedule: a cheap check of a new setup.
- `train.sbatch` cancels a run that makes no progress for 30 minutes. The W&B run is offline under
  `<marin checkout>/wandb`; `wandb sync` it from a login node.

`RELAY_CLUSTER=horizon` runs the same scripts on TACC Horizon; there `build_env.sh` runs as a one-node job that
reaches the internet through a login-side SOCKS tunnel.
