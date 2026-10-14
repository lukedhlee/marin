"""Teacher-forced NLL of rows under a native Grug checkpoint: forward only, no update, nothing saved but a small JSON.

Train/serve parity diagnostic for the online reflection backend: the same model construction, mesh, mixed precision,
router-bias application and loss as online_reflection.py, with reduction="none" so each supervised token's NLL can be
compared with vLLM's prompt logprobs on the same ids. Run under srun with the training layout.

    python -m experiments.june_tpu_67b_a2b.moe.forward_nll --init <native ckpt> --base <hf dir> --rows <jsonl> \
        --n 32 --seq-len 65536 --microbatch 16 --expert-axis 8 --out <json>
"""

# ruff: noqa: PLC0415 -- defer accelerator imports
import argparse
import dataclasses
import json
from pathlib import Path

BUCKETS = (0, 2048, 4096, 8192, 16384, 1 << 30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=65536)
    ap.add_argument("--microbatch", type=int, default=16)
    ap.add_argument("--expert-axis", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import jax
    import jmp
    import numpy as np
    import optax
    from haliax.partitioning import set_mesh
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from levanter.data.text.examples import GrugLmExample
    from levanter.distributed import DistributedConfig
    from levanter.grug.attention import AttentionMask
    from levanter.grug.sharding import _batch_axes, compact_grug_mesh

    from experiments.june_tpu_67b_a2b.moe.snowball_chat_recipe import read_base_hf_config, snowball_model_config_for_base
    from experiments.june_tpu_67b_a2b.moe.train import (
        _apply_qb_betas,
        apply_gpu_runtime_defaults,
        init_weights_only_from_checkpoint,
        initial_state,
    )

    rows = [json.loads(line) for line in Path(a.rows).read_text().splitlines() if line.strip()][: a.n]
    base_config = read_base_hf_config(a.base)
    model_config = dataclasses.replace(
        snowball_model_config_for_base(base_config, max_seq_len=a.seq_len), router_z_loss_coef=0.0
    )
    apply_gpu_runtime_defaults()
    DistributedConfig().initialize()
    mp = jmp.get_policy("p=f32,c=bfloat16")
    mesh = compact_grug_mesh(expert_axis_size=a.expert_axis, replica_axis_size=1, model_axis_size=1)
    axes = _batch_axes(mesh)
    size, length = a.microbatch, a.seq_len
    results = []
    with set_mesh(mesh):
        state = jax.jit(
            lambda: initial_state(model_config, optimizer=optax.sgd(0.0), mp=mp, key=jax.random.PRNGKey(0), ema_beta=None)
        )()
        state = init_weights_only_from_checkpoint(state, a.init, mesh=mesh, load_ema=False)
        params = _apply_qb_betas(state.params, state.pending_qb_betas)

        @jax.jit
        def per_position(params, batch):
            return mp.cast_to_compute(params).next_token_loss(
                batch.tokens, batch.loss_weight, mask=batch.attn_mask, reduction="none"
            )

        sharding = NamedSharding(mesh, P(axes, None))
        for offset in range(0, len(rows), size):
            subset = rows[offset : offset + size]
            tokens = np.full((size, length), 128001, dtype=np.int32)
            weights = np.zeros((size, length), dtype=np.float32)
            for index, row in enumerate(subset):
                n = len(row["ids"])
                tokens[index, :n] = row["ids"]
                weights[index, : n - 1] = row["loss"][1:]
            batch = GrugLmExample(
                jax.make_array_from_callback(tokens.shape, sharding, lambda i, v=tokens: v[i]),
                jax.make_array_from_callback(weights.shape, sharding, lambda i, v=weights: v[i]),
                AttentionMask.causal(),
            )
            nll = np.asarray(multihost_utils.process_allgather(per_position(params, batch), tiled=True))
            for index, row in enumerate(subset):
                # Position i predicts token i + 1; report by the target token's position.
                w = weights[index]
                per = nll[index] * w
                stats = {"sid": row["sid"], "tokens": int(w.sum()), "nll": float(per.sum() / w.sum())}
                for lo, hi in zip(BUCKETS, BUCKETS[1:]):
                    sel = w[max(lo - 1, 0) : hi - 1]
                    if sel.sum():
                        stats[f"pos_{lo}"] = [float((nll[index][max(lo - 1, 0) : hi - 1] * sel).sum()), int(sel.sum())]
                results.append(stats)
    if jax.process_index() == 0:
        total = sum(r["nll"] * r["tokens"] for r in results) / sum(r["tokens"] for r in results)
        Path(a.out).write_text(json.dumps({"init": a.init, "rows": len(results), "nll": total, "per_row": results}, indent=1))
        print(json.dumps({"init": a.init, "rows": len(results), "nll": round(total, 4)}))
    multihost_utils.sync_global_devices("forward_nll_done")


if __name__ == "__main__":
    main()
