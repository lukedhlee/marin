# Horizon <-> Jupiter SFT port pair (current recipe), 2026-09-30

**What this is.** It runs one identical short SFT arm on Horizon (TACC, GB200) and on Jupiter (JSC, GH200), so we
can judge whether Horizon trains Snowball SFT like Jupiter under the recipe we use now. The arm has 30 steps, and both
clusters use the same code, data, init and batches. The Horizon side is already running. A Jupiter Claude session
runs the Jupiter side from this file alone. Owner: the Horizon session; status in `~/briefs/train-port.STATUS.md`
on Horizon.

## The arm (identical on both sides)

| | value |
|---|---|
| code | this branch, `lukedhlee/horizon-snowball-sft-0921` on github.com/lukedhlee/marin. Training code frozen at **7db3fe6b9** (= `lukedhlee/vista-snowball-sft` tip fa26be983 + Horizon scripts + the stage + the driver). Later commits on this branch touch only this file. |
| base | Grug Datakit 09-21 = HF `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` @ b8c07f7d (39 shards, 134.2 GB, training template sha256 6f55d2ce…) |
| init | `import_snowball_hf --base_config_from_hf true --pending_from_router_bias true` (snowball_base.json sidecar; pending_qb_betas = −router_bias). The router bias stays frozen at 09-21's value (stage default). |
| template / format | 09-21 training template; datakit "think" rows (reasoning -> reasoning_content, enable_thinking true) |
| data | 2,048 Kimi SWE-smith traces = 30.9M 09-21 tokens = 30 token-derived steps. Private HF dataset `lukeleeai/snowball-kimi0921-pair` @ **2883a32db3232284bae7a65130a30ded0b696519**, `train-00000-of-00001.parquet` sha256 **7683b2ff5177159a099939ba6ea9578130d97f7175e401048da5738c18d1a094**. Built by OpenThoughts-Agent `data/swesmith/kimi_0921_convert.py` (seed 20260930) from the public `open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k` @ 1b87a9cf. |
| stage | `kimi0921_pair` (`vista_snowball_chat.py`) |
| layout | 16 sequences x 65,536 tokens per step, 4 nodes x 4 one-GPU ranks, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` (the relay/bespoke settings) |
| optimizer | stage AdamH with `SNOWBALL_LR=3e-4`, `SNOWBALL_WARMUP=2` (steps), one cosine over 30 steps, seed 0 |
| driver | `experiments/june_tpu_67b_a2b/moe/pair_kimi0921.sh`: data download + sha256 check -> prep (cache, 1 node) + import (4 nodes; skipped when the init sidecar exists) -> run (4 nodes, 30 steps) -> W&B sync + a per-step loss TSV |

## Run the Jupiter side (Jupiter login node)

Cost: ~1.5–2 node-h (prep 1 node x ~15 min; the import is skipped because the Bespoke/relay init
`/e/data1/mmlaion/lee27/snowball-sft/experiments/snowball-base-inits/init-dk0921-step0` already exists, made by the same
importer code at 3a4a29509, unchanged since; run 4 nodes x ~25–40 min). Jupiter is reserved 2026-09-30 04:00 PT ->
10-02 04:00 PT. Run before or after that window.

```bash
C=/e/project1/transfernetx/lee27/code
# a separate worktree: marin-sft itself stays on its branch for the other SFT work
git -C $C/marin-sft fetch https://github.com/lukedhlee/marin.git lukedhlee/horizon-snowball-sft-0921
git -C $C/marin-sft worktree add --detach $C/marin-sft-pair FETCH_HEAD
git -C $C/marin-sft-pair diff --quiet 7db3fe6b9 HEAD -- experiments lib && echo "training code = 7db3fe6b9"
# the env is the existing one (the uv.lock is unchanged since 09-16); the driver defaults to it
tmux new -d -s pair_kimi0921 "CLUSTER=jupiter REPLICA=r1 SBATCH_ACCOUNT=laionize bash $C/marin-sft-pair/experiments/june_tpu_67b_a2b/moe/pair_kimi0921.sh; sleep 3600"
tail -f /e/data1/mmlaion/lee27/snowball-sft/logs/kimi0921_pair/pair_jupiter.log      # ends with PAIR_DONE
```

The driver's Jupiter defaults, each overridable by env:
- `SNOWBALL_SCRATCH=/e/data1/mmlaion/lee27/snowball-sft`
- `MARIN_PYTHON=$C/envs/marin-grug-sft/bin/python`
- `BASE_0921=/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921`
- `SNOWBALL_INIT=<scratch>/experiments/snowball-base-inits/init-dk0921-step0`
- `HF_CLI=$C/envs/snowball-v2/bin/hf` (the dataset is private to lukeleeai, so the HF token must be Luke's; export
  `HF_TOKEN` if the download returns 401)
- `WANDB_CLI=$C/envs/snowball-v2/bin/wandb`
- `KEYS=/e/fscratch/reformo/lee27/keys/secrets.env` (holds `WANDB_API_KEY`)

Outputs:
- `logs/kimi0921_pair/{pair_jupiter.log, prep.*.log, loss_snowball-kimi0921-pair-jupiter-r1.tsv}`
- the run log `logs/snowball-kimi0921_pair.<job>.log`
- the offline W&B run under `$C/marin-sft-pair/wandb/`

When done, delete the run's native checkpoint (`experiments/snowball-kimi0921-pair/*/checkpoints`, hundreds of GB).
Nothing is exported.

If a step fails: fix only an environment problem (paths, account, keys) and rerun. The driver skips finished steps.
Do not change the recipe. Report the failure (log path and last lines) to the Mac session, which relays to the Horizon session.

## W&B

Entity `lukedhlee-marin`, project **`horizon-jupiter-sft-pair`**. Run names (= run ids):
- `snowball-kimi0921-pair-horizon-r1`
- `snowball-kimi0921-pair-horizon-r2` (Horizon rerun, measures noise)
- `snowball-kimi0921-pair-jupiter-r1`

Metric: `train/loss` at W&B steps 0..29.

## Pass rule (written 2026-09-29 22:45 PT, before any result)

Let Δ_t = loss_Horizon-r1(t) − loss_Jupiter-r1(t) for t = 0..29 (the same batches by construction). Let σ = the mean
over t = 1..29 of |loss_Horizon-r1(t) − loss_Horizon-r2(t)| (Horizon rerun noise). PASS iff all three hold:
1. |Δ_0| ≤ 0.005 (step 0 runs at lr 0 on the same weights with the frozen 09-21 bias: forward parity);
2. mean over t = 1..29 of |Δ_t| ≤ max(3σ, 0.003);
3. max over t = 1..29 of |Δ_t| ≤ 0.02.

Background: on the old recipe (Stage-3 import with zeroed pending_qb_betas and a per-batch router bias; Kimi, 96
steps) Horizon ran a systematic ~0.010 below Jupiter against a rerun noise of 0.0009, and would fail clause 2. The
trained models still matched on held-out NLL (0.3698 vs 0.374). This pair tests whether the frozen, non-zero bias of
the current recipe removes that offset.

## Horizon side: done (2026-09-29 23:05 PT)

Both replicas ran end to end with this driver and are on W&B:
- r1 loss 0.3774 / 0.2987 / 0.2774 at steps 0 / 9 / 29;
- rerun noise σ = 0.00053, so the clause-2 threshold is 0.003.

As a setup check for the Jupiter side, the prep log should read `cache_tokens=30922127 cache_examples=2048`, and the
launcher plan should read `steps = 30`, `layout = 16 x 65536 ... on 4 nodes (16 ranks); warmup 2`, `lr = 3e-4`.
