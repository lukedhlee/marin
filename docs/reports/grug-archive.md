# Grug Archive: Experiments and Snapshots

This file is the paper trail for grug experiments.

## Principles

- `experiments/grug/base/` is the canonical template.
- Speedrun files are exploratory and may be deleted after upstreaming.
- Prefer deletion over long-term maintenance of stale experiment code.

## Entry Template

```text
### <experiment-id>
- Path: <repo-relative-path>
- Origin: base | moe | <source variant; omit for entries predating the base template>
- Introduced: <commit-sha>
- Last known-good: <commit-sha; superseded and deleted entries only>
- Status: active | superseded | deleted
- Purpose: <one line>
- Superseded by: <path or commit; optional>
- Diff: <CI-posted grug diff comment, or local report path; optional>
- Issue: <url/id; optional>
```

Active entries track `main`, so they omit `Last known-good` rather than carry a SHA that
goes stale on the next commit.

## Experiments

### snowball-relay-student-loss-01
- Path: `experiments/june_tpu_67b_a2b/moe/`
- Origin: matched original-Relay+Kimi FP32/capacity2 run at `3d89809e55`
- Status: active
- Purpose: keep all baseline tokens and teacher targets, add loss weight0.1 to recorded student-response spans, and compare the final723-step model with the completed student-masked control.
- Diff: `lukedhlee/relay-student01-20261025` versus `3d89809e55`.

### snowball-original-relay-kimi-capacity2
- Path: `experiments/june_tpu_67b_a2b/moe/`
- Origin: capacity2 vanilla+Kimi experiment at `96f54493fc`
- Status: active
- Purpose: original Relay on the same 1,487 CalibForge tasks and exact 4,392 Kimi rows as vanilla, with FP32 masters and expert capacity2; three passes, final-checkpoint evaluation.
- Diff: `lukedhlee/relay-kimi-capacity2-20261021` versus `96f54493fc`.

### snowball-sft-capacity2
- Path: `experiments/june_tpu_67b_a2b/moe/`
- Origin: existing June MoE experiment, FP32 SFT branch at `81157cb732`
- Status: active
- Purpose: increase expert-parallel dispatch capacity from 1.0 to 2.0 for new SFT runs after observing 11–14% assignment drops; GPU memory, throughput, and remaining overflow require measurement.
- Diff: `lukedhlee/snowball-capacity2-20261021` versus `81157cb732`.

### grug-base-template
- Path: `experiments/grug/base/`
- Origin: base
- Introduced: 8d752a775
- Status: active
- Purpose: canonical grug template (model/train/launch).

### grug-moe
- Path: `experiments/grug/moe/`
- Origin: base
- Introduced: 9181fd753
- Status: active
- Purpose: canonical Mixture-of-Experts variant; carries its own model, optimizer, train loop, and launch wiring so it can iterate independently of the dense template.
- Issue: https://github.com/marin-community/marin/pull/3046

### grug-moe-hero-ep
- Path: `experiments/grug/moe_hero_ep/`
- Origin: `moe_hero_fsdp` at PR 7876
- Introduced: b0d20062a
- Status: active
- Purpose: one-rack GB200 EP64 throughput and MFU baseline.
- Issue: https://github.com/marin-community/marin/issues/7279

### grugformer-vs-hackable-125m
- Path: `experiments/speedrun/grugformer_vs_hackable_125m/grugformer_vs_hackable_125m.py`
- Introduced: 5efe76834
- Last known-good: 5efe76834
- Status: deleted
- Purpose: historical head-to-head comparison.
- Superseded by: 8d752a775 — template-first workflow centered on `experiments/grug/base/`.
