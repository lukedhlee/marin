# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""TailSFT filtering for the grug SFT trainer (arXiv 2608.25756, Algorithm 1).

Standard SFT keeps raising the likelihood of every demonstration, including the ones the model already
produces, which drains probability from the other valid responses RL later needs to find. TailSFT drops the
``fraction`` of documents in each global batch whose length-normalised loss has fallen the most below the
same document's loss under the initial checkpoint, so the gradient concentrates on the tail.

Everything here works on raw per-position arrays and is written to run INSIDE a ``jax.shard_map`` over the
batch axes (the same layout ``levanter.grug.loss`` uses): every device holds its rows of the packed batch,
per-document sums are ``psum``-ed to replicated ``(num_docs,)`` vectors, the rank/threshold is computed
redundantly on every device from identical inputs, and the kept-token reduction is ``psum``-ed back to a
scalar. The pure functions take a ``psum`` callable so a single-device unit test can pass the identity.

Document identity is the packed segment id, which the prepacked dataset sets to the GLOBAL cache row index
(``levanter/data/packing.py`` ``GreedyPrepackedDataset``: ``global_doc_idx = dr.start + doc_idx``), so a
``(num_docs,)`` reference vector indexed by segment id needs no join table. Padding carries ``-1``.

``per_pos`` is what ``next_token_loss(..., reduction="none")`` returns: the per-position loss ALREADY
multiplied by ``loss_weight`` (and including the logit z-loss term, see
``kernels/pallas/fused_cross_entropy_loss/api.py::_apply_reduction``). The "mean" reduction the trainer
uses is ``psum(sum(per_pos)) / psum(sum(loss_weight))``; the filtered loss is the same ratio over the kept
tokens only. ``fraction == 0`` keeps every token, but the trainer never routes through here in that case:
the flag-off path executes the untouched ``reduction="mean"`` lines.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

PsumFn = Callable[[jax.Array], jax.Array]

# Every scalar ``tail_filtered_loss`` reports, in a fixed order, so a ``shard_map`` caller can declare the
# replicated out_specs for the stats dict statically.
STAT_KEYS: tuple[str, ...] = (
    "tail/present_docs",
    "tail/scorable_docs",
    "tail/dropped_docs",
    "tail/target_drop",
    "tail/threshold_margin",
    "tail/mean_margin",
    "tail/mean_doc_loss",
    "tail/loss_unfiltered",
    "tail/kept_token_frac",
    "tail/fraction",
)


def _identity(x: jax.Array) -> jax.Array:
    return x


def tail_schedule(step: jax.Array | int, fraction: float, ramp_steps: int) -> jax.Array:
    """Filter fraction at ``step``: static ``fraction``, or a linear ramp 0 -> fraction over ``ramp_steps``."""
    f = jnp.asarray(fraction, dtype=jnp.float32)
    if ramp_steps <= 0:
        return f
    prog = jnp.clip(jnp.asarray(step, dtype=jnp.float32) / float(ramp_steps), 0.0, 1.0)
    return f * prog


def document_sums(
    per_pos: jax.Array,
    weight: jax.Array,
    segment_ids: jax.Array,
    num_docs: int,
    *,
    psum: PsumFn = _identity,
) -> tuple[jax.Array, jax.Array]:
    """Per-document (sum of weighted loss, sum of loss weight), replicated over the batch axes.

    ``per_pos``/``weight``/``segment_ids`` are the LOCAL shards ``(b, S)``; the result is ``(num_docs,)``
    float32 after ``psum``. Padding (segment id ``-1``) contributes nothing. A document's length-normalised
    loss is ``num / den`` wherever ``den > 0``.
    """
    seg = segment_ids.reshape(-1).astype(jnp.int32)
    valid = seg >= 0
    seg_safe = jnp.where(valid, seg, 0)
    pp = jnp.where(valid, per_pos.reshape(-1).astype(jnp.float32), 0.0)
    w = jnp.where(valid, weight.reshape(-1).astype(jnp.float32), 0.0)
    num = jax.ops.segment_sum(pp, seg_safe, num_segments=num_docs)
    den = jax.ops.segment_sum(w, seg_safe, num_segments=num_docs)
    return psum(num), psum(den)


def tail_keep_mask(
    doc_num: jax.Array,
    doc_den: jax.Array,
    ref_loss: jax.Array,
    fraction: jax.Array | float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Which documents keep their gradient this step.

    ``margin = current length-normalised loss - reference loss``; the ``floor(fraction * M)`` documents with
    the smallest margin among the ``M`` documents that are present in the batch AND have a finite reference
    are dropped. Documents without a reference (NaN in ``ref_loss``: never scored) are never dropped.
    Returns the replicated ``(num_docs,)`` boolean keep mask and scalar diagnostics.
    """
    num_docs = doc_num.shape[0]
    present = doc_den > 0
    ell = jnp.where(present, doc_num / jnp.where(present, doc_den, 1.0), 0.0)
    ref = jnp.asarray(ref_loss, dtype=jnp.float32)
    scorable = present & jnp.isfinite(ref)
    margin = jnp.where(scorable, ell - ref, jnp.inf)
    m = jnp.sum(scorable).astype(jnp.int32)
    k = jnp.floor(jnp.asarray(fraction, dtype=jnp.float32) * m.astype(jnp.float32)).astype(jnp.int32)
    k = jnp.clip(k, 0, jnp.maximum(m - 1, 0))  # never drop every scorable document
    ordered = jnp.sort(margin)  # +inf (not scorable) sorts last
    kth = ordered[jnp.clip(k - 1, 0, num_docs - 1)]
    thresh = jnp.where(k > 0, kth, -jnp.inf)
    drop = scorable & (margin <= thresh) & (k > 0)
    keep = jnp.logical_not(drop)
    n_scorable = jnp.maximum(m, 1).astype(jnp.float32)
    stats = {
        "tail/present_docs": jnp.sum(present).astype(jnp.float32),
        "tail/scorable_docs": m.astype(jnp.float32),
        "tail/dropped_docs": jnp.sum(drop).astype(jnp.float32),
        "tail/target_drop": k.astype(jnp.float32),
        "tail/threshold_margin": thresh,
        "tail/mean_margin": jnp.sum(jnp.where(scorable, margin, 0.0)) / n_scorable,
        "tail/mean_doc_loss": jnp.sum(jnp.where(present, ell, 0.0)) / jnp.maximum(jnp.sum(present), 1).astype(jnp.float32),
    }
    return jax.lax.stop_gradient(keep), {k_: jax.lax.stop_gradient(v) for k_, v in stats.items()}


def filtered_mean(
    per_pos: jax.Array,
    weight: jax.Array,
    segment_ids: jax.Array,
    keep_doc: jax.Array,
    *,
    psum: PsumFn = _identity,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(filtered loss, unfiltered loss, kept token fraction)`` from LOCAL shards and the replicated keep mask.

    The unfiltered value is exactly the trainer's ``reduction="mean"`` quantity, so the two are comparable.
    """
    seg = segment_ids.reshape(-1).astype(jnp.int32)
    valid = seg >= 0
    seg_safe = jnp.where(valid, seg, 0)
    keep_tok = jax.lax.stop_gradient(valid & jnp.take(keep_doc, seg_safe))
    pp = per_pos.reshape(-1).astype(jnp.float32)
    w = weight.reshape(-1).astype(jnp.float32)
    loss_sum = psum(jnp.sum(jnp.where(keep_tok, pp, 0.0)))
    w_kept = psum(jnp.sum(jnp.where(keep_tok, w, 0.0)))
    loss_all = psum(jnp.sum(pp))
    w_all = psum(jnp.sum(w))
    filtered = jnp.where(w_kept > 0, loss_sum / jnp.where(w_kept > 0, w_kept, 1.0), 0.0)
    unfiltered = jnp.where(w_all > 0, loss_all / jnp.where(w_all > 0, w_all, 1.0), 0.0)
    kept_frac = jnp.where(w_all > 0, w_kept / jnp.where(w_all > 0, w_all, 1.0), 0.0)
    return filtered, unfiltered, kept_frac


def tail_filtered_loss(
    per_pos: jax.Array,
    weight: jax.Array,
    segment_ids: jax.Array,
    ref_loss: jax.Array,
    fraction: jax.Array | float,
    *,
    psum: PsumFn = _identity,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The whole TailSFT step on local shards: per-document sums -> keep mask -> filtered mean.

    Returns ``(loss, stats)``; ``stats`` are replicated scalars for logging, including the unfiltered mean
    under ``tail/loss_unfiltered`` and the kept-token fraction under ``tail/kept_token_frac``.
    """
    num_docs = ref_loss.shape[0]
    doc_num, doc_den = document_sums(per_pos, weight, segment_ids, num_docs, psum=psum)
    keep_doc, stats = tail_keep_mask(doc_num, doc_den, ref_loss, fraction)
    loss, unfiltered, kept_frac = filtered_mean(per_pos, weight, segment_ids, keep_doc, psum=psum)
    stats = dict(stats)
    stats["tail/loss_unfiltered"] = jax.lax.stop_gradient(unfiltered)
    stats["tail/kept_token_frac"] = jax.lax.stop_gradient(kept_frac)
    stats["tail/fraction"] = jnp.asarray(fraction, dtype=jnp.float32)
    assert tuple(stats) == STAT_KEYS, (tuple(stats), STAT_KEYS)
    return loss, stats


def load_reference_losses(path: str, num_docs: int) -> np.ndarray:
    """Load the ``(num_docs,)`` float32 reference vector written by the scoring pass; NaN = never scored."""
    ref = np.load(path)
    if ref.ndim != 1 or ref.shape[0] != num_docs:
        raise ValueError(f"reference losses at {path} have shape {ref.shape}; the cache has {num_docs} documents")
    ref = ref.astype(np.float32)
    n_scored = int(np.isfinite(ref).sum())
    if n_scored == 0:
        raise ValueError(f"reference losses at {path} contain no finite entry")
    return ref
