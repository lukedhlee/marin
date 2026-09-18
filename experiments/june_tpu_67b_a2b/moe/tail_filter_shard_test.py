# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""4-device CPU check of the shard_map pattern train.py uses around tail_filter: values and gradients
through the psums must equal the single-device computation."""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from experiments.june_tpu_67b_a2b.moe import tail_filter as tf
from experiments.june_tpu_67b_a2b.moe.tail_filter_test import make_batch


def main():
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices, ("data",))
    rng = np.random.default_rng(1)
    per_pos, weight, seg, used = make_batch(rng, b=8, s=48, num_docs=23, docs_per_row=3)
    n = 23
    ref = rng.uniform(0.2, 2.5, size=n).astype(np.float32)
    ref[3] = np.nan
    fraction = 0.3

    row = NamedSharding(mesh, P("data"))
    rep = NamedSharding(mesh, P())
    pp = jax.device_put(jnp.asarray(per_pos), row)
    w = jax.device_put(jnp.asarray(weight), row)
    sg = jax.device_put(jnp.asarray(seg), row)
    rf = jax.device_put(jnp.asarray(ref), rep)
    fr = jax.device_put(jnp.asarray(fraction, dtype=jnp.float32), rep)

    def psum(x):
        return jax.lax.psum(x, "data")

    def local(a, b, c, r, f):
        return tf.tail_filtered_loss(a, b, c, r, f, psum=psum)

    stats_specs = {k: P() for k in tf.STAT_KEYS}
    sharded = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(P("data"), P("data"), P("data"), P(), P()),
                                    out_specs=(P(), stats_specs), check_vma=False))
    loss_s, stats_s = sharded(pp, w, sg, rf, fr)
    loss_1, stats_1 = jax.jit(tf.tail_filtered_loss)(jnp.asarray(per_pos), jnp.asarray(weight), jnp.asarray(seg), jnp.asarray(ref), fraction)
    assert abs(float(loss_s) - float(loss_1)) < 1e-5, (float(loss_s), float(loss_1))
    for k in tf.STAT_KEYS:
        a, b = float(stats_s[k]), float(stats_1[k])
        assert (np.isinf(a) and np.isinf(b)) or abs(a - b) < 1e-4, (k, a, b)

    # gradients wrt per_pos: single device vs through shard_map (+ psum transposes with check_vma=False)
    g1 = np.asarray(jax.grad(lambda x: tf.tail_filtered_loss(x, jnp.asarray(weight), jnp.asarray(seg), jnp.asarray(ref), fraction)[0])(jnp.asarray(per_pos)))
    def sharded_loss(x):
        return sharded(x, w, sg, rf, fr)[0]
    gs = np.asarray(jax.grad(sharded_loss)(pp))
    assert np.allclose(g1, gs, atol=1e-6), (np.abs(g1 - gs).max())
    # the plain mean's gradient through the same pattern, as a control for the psum transpose scale
    def mean_local(a, b, c, *, psum):
        return psum(jnp.sum(a)) / psum(jnp.sum(b))
    mean_sharded = jax.jit(jax.shard_map(lambda a, b, c: mean_local(a, b, c, psum=psum), mesh=mesh,
                                         in_specs=(P("data"), P("data"), P("data")), out_specs=P(), check_vma=False))
    gm = np.asarray(jax.grad(lambda x: mean_sharded(x, w, sg))(pp))
    gm1 = np.asarray(jax.grad(lambda x: jnp.sum(x) / jnp.sum(jnp.asarray(weight)))(jnp.asarray(per_pos)))
    assert np.allclose(gm, gm1, atol=1e-7), "control: plain-mean gradient through shard_map differs"
    print(f"sharded loss {float(loss_s):.5f} == single {float(loss_1):.5f}; dropped {int(stats_s['tail/dropped_docs'])}; "
          f"grad max|diff| {np.abs(g1 - gs).max():.2e}; devices {len(devices)}")
    print("TAIL_FILTER_SHARD_TEST_OK")


if __name__ == "__main__":
    main()
