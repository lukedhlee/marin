import dataclasses
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

path = Path(__file__).parents[2] / "experiments/june_tpu_67b_a2b/moe/online_reflection.py"
spec = importlib.util.spec_from_file_location("online_reflection_standalone", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@dataclasses.dataclass
class State:
    params: object
    opt_state: object
    step: int


def test_accumulated_global_token_mean_matches_direct_and_preserves_adam():
    optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adamw(0.01, b1=0.9, b2=0.98, weight_decay=0.1))
    params = jnp.array([0.2, 0.4])
    state = State(params, optimizer.init(params), 0)
    direct_state = State(params, optimizer.init(params), 0)
    loss_grad = jax.value_and_grad(lambda p, x, y, mask: jnp.sum((x @ p - y) ** 2 * mask) / jnp.sum(mask))
    for iteration in range(2):
        x = jnp.arange(10).reshape(5, 2).astype(jnp.float32) + iteration
        y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = jnp.array([1.0, 0.0, 1.0, 1.0, 1.0])
        _, direct_gradient = loss_grad(direct_state.params, x, y, mask)
        accumulated = jnp.zeros_like(state.params)
        for start, end in ((0, 2), (2, 5)):
            _, grad = loss_grad(state.params, x[start:end], y[start:end], mask[start:end])
            accumulated = module.weighted_add(accumulated, grad, float(mask[start:end].sum() / mask.sum()))
        np.testing.assert_allclose(accumulated, direct_gradient, rtol=1e-6)
        state = module.apply_optimizer_update(state, state.params, accumulated, optimizer, 4)
        direct_state = module.apply_optimizer_update(direct_state, direct_state.params, direct_gradient, optimizer, 4)
        np.testing.assert_allclose(state.params, direct_state.params, rtol=1e-6)
        for accumulated_leaf, direct_leaf in zip(
            jax.tree.leaves(state.opt_state), jax.tree.leaves(direct_state.opt_state), strict=True
        ):
            np.testing.assert_allclose(accumulated_leaf, direct_leaf, rtol=1e-6)
        assert state.step == iteration + 1
    # Distinguish true Adam continuation from resetting moments after every rollout.
    fresh = State(state.params, optimizer.init(state.params), 0)
    continued = module.apply_optimizer_update(state, state.params, jnp.array([-0.3, 0.1]), optimizer, 4)
    restarted = module.apply_optimizer_update(fresh, fresh.params, jnp.array([-0.3, 0.1]), optimizer, 4)
    assert not np.allclose(continued.params, restarted.params)
