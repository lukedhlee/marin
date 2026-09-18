# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU unit test for tail_filter: per-document sums, the drop rule and the filtered mean against numpy."""
import numpy as np
import jax
import jax.numpy as jnp

from experiments.june_tpu_67b_a2b.moe import tail_filter as tf

jax.config.update("jax_platforms", "cpu")


def make_batch(rng, b=4, s=64, num_docs=17, docs_per_row=3):
    seg = -np.ones((b, s), dtype=np.int32)
    weight = np.zeros((b, s), dtype=np.float32)
    used = 0
    for r in range(b):
        pos = 0
        for _ in range(docs_per_row):
            length = rng.integers(6, 18)
            if pos + length > s or used >= num_docs:
                break
            seg[r, pos:pos + length] = used
            # completion tokens only: mask the first third of each doc (the prompt)
            weight[r, pos + length // 3:pos + length] = 1.0
            pos += length
            used += 1
    raw_loss = rng.uniform(0.1, 3.0, size=(b, s)).astype(np.float32)
    per_pos = raw_loss * weight  # what reduction="none" returns
    return per_pos, weight, seg, used


def numpy_reference(per_pos, weight, seg, ref, fraction):
    n = ref.shape[0]
    num = np.zeros(n); den = np.zeros(n)
    for d in range(n):
        m = seg == d
        num[d] = per_pos[m].sum(); den[d] = weight[m].sum()
    present = den > 0
    ell = np.where(present, num / np.where(present, den, 1), 0)
    scorable = present & np.isfinite(ref)
    margin = np.where(scorable, ell - ref, np.inf)
    m = scorable.sum()
    k = int(np.floor(fraction * m))
    k = min(k, max(m - 1, 0))
    drop = np.zeros(n, bool)
    if k > 0:
        order = np.argsort(margin)
        drop[order[:k]] = True
    keep_tok = (seg >= 0) & ~drop[np.where(seg >= 0, seg, 0)]
    loss = per_pos[keep_tok].sum() / weight[keep_tok].sum()
    unf = per_pos.sum() / weight.sum()
    return loss, unf, drop, k


def main():
    rng = np.random.default_rng(0)
    per_pos, weight, seg, used = make_batch(rng)
    n = 17
    ref = rng.uniform(0.2, 2.5, size=n).astype(np.float32)
    ref[used - 1] = np.nan  # one present doc without a reference: never dropped
    ref[n - 1] = np.nan  # an absent doc

    num, den = tf.document_sums(jnp.asarray(per_pos), jnp.asarray(weight), jnp.asarray(seg), n)
    num_np = np.zeros(n); den_np = np.zeros(n)
    for d in range(n):
        m = seg == d
        num_np[d] = per_pos[m].sum(); den_np[d] = weight[m].sum()
    assert np.allclose(np.asarray(num), num_np, atol=1e-5), "per-doc numerators"
    assert np.allclose(np.asarray(den), den_np, atol=1e-5), "per-doc denominators"

    for fraction in (0.0, 0.25, 0.5, 0.9, 1.0):
        loss, stats = jax.jit(tf.tail_filtered_loss)(
            jnp.asarray(per_pos), jnp.asarray(weight), jnp.asarray(seg), jnp.asarray(ref), fraction
        )
        loss_np, unf_np, drop_np, k_np = numpy_reference(per_pos, weight, seg, ref, fraction)
        assert abs(float(loss) - loss_np) < 1e-5, (fraction, float(loss), loss_np)
        assert abs(float(stats["tail/loss_unfiltered"]) - unf_np) < 1e-5
        assert int(stats["tail/dropped_docs"]) == k_np == drop_np.sum(), (fraction, int(stats["tail/dropped_docs"]), k_np)
        # the doc without a reference is present and must never be dropped
        keep_doc, _ = tf.tail_keep_mask(num, den, jnp.asarray(ref), fraction)
        assert bool(keep_doc[used - 1])
        assert np.array_equal(np.asarray(~keep_doc), drop_np)
        if fraction == 0.0:
            assert abs(float(loss) - unf_np) < 1e-6
            assert float(stats["tail/kept_token_frac"]) == 1.0
        print(f"fraction {fraction}: loss {float(loss):.4f} unfiltered {unf_np:.4f} dropped {k_np} "
              f"of {int(stats['tail/scorable_docs'])} scorable, kept tokens {float(stats['tail/kept_token_frac']):.3f}")

    # gradient flows only through kept tokens
    def f(pp):
        return tf.tail_filtered_loss(pp, jnp.asarray(weight), jnp.asarray(seg), jnp.asarray(ref), 0.5)[0]
    g = np.asarray(jax.grad(f)(jnp.asarray(per_pos)))
    _, _, drop_np, _ = numpy_reference(per_pos, weight, seg, ref, 0.5)
    dropped_tok = (seg >= 0) & drop_np[np.where(seg >= 0, seg, 0)]
    assert np.all(g[dropped_tok] == 0), "dropped tokens must carry no gradient"
    assert np.all(g[(weight > 0) & ~dropped_tok] > 0), "kept completion tokens must carry gradient"
    # ramp schedule
    assert float(tf.tail_schedule(0, 0.25, 10)) == 0.0
    assert abs(float(tf.tail_schedule(5, 0.25, 10)) - 0.125) < 1e-6
    assert float(tf.tail_schedule(50, 0.25, 10)) == 0.25
    assert float(tf.tail_schedule(3, 0.25, 0)) == 0.25
    print("TAIL_FILTER_TEST_OK")


if __name__ == "__main__":
    main()
