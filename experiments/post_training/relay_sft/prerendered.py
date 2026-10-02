# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training rows that arrive tokenized with a per-token loss mask.

A relay row is an episode in which the student model played the first turns and the teacher finished; only the
teacher's turns are trained. That mask is not a chat template's assistant mask, so the rows are rendered upstream
(OpenThoughts-Agent ``data/relay/sft``) with the base's own serving template and arrive as ``ids`` plus ``loss``
(``loss[t] = 1``: token ``t`` is a trained target). ``PrerenderedChatFormat`` is a ``ChatLmDatasetFormat`` whose
preprocessor copies them into the cache fields the chat preprocessor writes, so packing, the shift of the mask onto
the predicting position and the loss are the chat path's, and nothing is re-tokenized.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from levanter.data._preprocessor import BatchProcessor
from levanter.data.text.formats import ChatLmDatasetFormat, LmDatasetFormatBase
from levanter.tokenizers import MarinTokenizer

ROW_ADAPTER = "prerendered_ids_loss_v1"


class PrerenderedRowProcessor(BatchProcessor[dict, dict]):
    """Copy ``ids`` / ``loss`` into ``input_ids`` / ``assistant_masks`` after checking each row.

    A row must have equal-length ids and loss, loss values 0 or 1, ids inside the vocabulary, an untrained BOS first,
    at least one trained token, and at most ``max_tokens`` ids: the packer's left slice would silently drop the
    trained tail of a longer row.
    """

    def __init__(self, *, ids_field: str, loss_field: str, max_tokens: int, vocab_size: int, bos_id: int):
        self.ids_field = ids_field
        self.loss_field = loss_field
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.bos_id = bos_id

    def __call__(self, batch: Sequence[dict]) -> Sequence[dict]:
        return [self._copy_row(row) for row in batch]

    def _copy_row(self, row: dict) -> dict:
        ids = np.asarray(row[self.ids_field], dtype=np.int64)
        loss = np.asarray(row[self.loss_field], dtype=np.int64)
        name = row.get("id")
        if ids.ndim != 1 or ids.shape != loss.shape:
            raise ValueError(f"row {name!r}: ids {ids.shape} and loss {loss.shape} differ")
        if not 0 < ids.size <= self.max_tokens:
            raise ValueError(f"row {name!r} has {ids.size} tokens; rows must hold 1 to {self.max_tokens}")
        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError(f"row {name!r} has ids outside [0, {self.vocab_size})")
        if not np.isin(loss, (0, 1)).all():
            raise ValueError(f"row {name!r} has loss values other than 0 and 1")
        if ids[0] != self.bos_id or loss[0] != 0:
            raise ValueError(f"row {name!r} must start with an untrained BOS ({self.bos_id})")
        if not loss.any():
            raise ValueError(f"row {name!r} trains no token")
        return {"input_ids": ids.astype(np.int32), "assistant_masks": loss.astype(np.int32)}

    @property
    def output_exemplar(self):
        return {"input_ids": np.zeros((0,), dtype=np.int32), "assistant_masks": np.zeros((0,), dtype=np.int32)}

    @property
    def num_cpus(self) -> int:
        return 1

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "row_adapter": ROW_ADAPTER,
            "ids_field": self.ids_field,
            "loss_field": self.loss_field,
            "max_tokens": self.max_tokens,
            "vocab_size": self.vocab_size,
            "bos_id": self.bos_id,
        }


@LmDatasetFormatBase.register_subclass("prerendered_chat")
@dataclass(frozen=True)
class PrerenderedChatFormat(ChatLmDatasetFormat):
    """Rows rendered and masked upstream; ``mask_user_turns`` must stay True, since the row's loss is the mask."""

    ids_field: str = "ids"
    loss_field: str = "loss"
    max_tokens: int = 65_536

    def build_preprocessor(
        self, tokenizer: MarinTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
    ) -> BatchProcessor[dict, dict]:
        del enforce_eos, enforce_bos  # rows carry their own BOS and end-of-turn tokens
        if not self.mask_user_turns:
            raise ValueError("PrerenderedChatFormat needs mask_user_turns=True: the row's loss is the mask")
        if tokenizer.bos_token_id is None:
            raise ValueError(f"tokenizer {tokenizer.name_or_path} has no BOS token")
        return PrerenderedRowProcessor(
            ids_field=self.ids_field,
            loss_field=self.loss_field,
            max_tokens=self.max_tokens,
            vocab_size=len(tokenizer),
            bos_id=tokenizer.bos_token_id,
        )
