# Relay SFT on Grug 67B-A2B 09-21 (pointer)

The relay SFT recipe trains 09-21 on prerendered relay rows (token ids + loss mask; only the Qwen3.8-27B teacher's turns
carry loss). In this branch: the stages (`_relay_stage` and the `relay_*` entries in
`experiments/june_tpu_67b_a2b/moe/vista_snowball_chat.py`: prerendered, no chat template, frozen router bias, 09-21
tokenizer pinned), the HF import with `pending_qb_betas = -router_bias` (`import_snowball_hf.py`,
`horizon_snowball_import.sbatch`), and the Horizon train / export sbatch files. Recipe: LR 3e-4, warmup 5 %, one cosine
over 3 passes, 16 x 65,536 tokens per step on 4 GB200 nodes.

End-to-end runbook (generation, rows, this SFT, evals), artifacts and findings: `data/relay/HANDOFF.md` in
`lukedhlee/OpenThoughts-Agent` (branch `lukedhlee/vista-moe-grpo-30b`). Models: `laion/snowball-67b-a2b-relay-sft-*`;
traces: `laion/calibforge-relay-traces`.
