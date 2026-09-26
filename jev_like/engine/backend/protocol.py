from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from jev_like.engine.storage import PagedKVStore


@dataclass(frozen=True, slots=True)
class PagedBatch:
    """Ragged Query 与 Paged KV Page Table 的 GPU 输入。"""

    token_data: torch.Tensor
    qo_indptr: torch.Tensor
    page_indptr: torch.Tensor
    page_indices: torch.Tensor
    last_page_len: torch.Tensor
    old_seq_lens: torch.Tensor
    append_batch_indices: torch.Tensor
    append_positions: torch.Tensor


class AttentionBackend(Protocol):
    """Paged Backbone Runner 依赖的最小 Attention Backend。"""

    @property
    def device(self) -> torch.device: ...
    def run(self, batch: PagedBatch, store: PagedKVStore) -> tuple[torch.Tensor, ...]: ...
