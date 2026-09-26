from __future__ import annotations

from dataclasses import dataclass

import torch

from jev_like.engine.backend import AttentionBackend, PagedBatch
from jev_like.engine.core.batch_plan import BatchPlan, Stage
from jev_like.engine.kv import KVCacheManager
from jev_like.engine.storage import PagedKVStore


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Paged ModelRunner 的纯计算结果。"""

    stage: Stage
    node_ids: tuple[int, ...]
    representations: tuple[torch.Tensor | None, ...]
    seq_lens: tuple[int, ...]


class ModelRunner:
    """将 Ragged BatchPlan 转换为 Paged Backbone 计算。"""

    def __init__(self, backend: AttentionBackend, kv_manager: KVCacheManager, kv_store: PagedKVStore) -> None:
        self.backend = backend
        self.kv_manager = kv_manager
        self.kv_store = kv_store

    def run(self, plan: BatchPlan) -> BatchResult:
        old_lens = [self.kv_manager.seq_len(prefix_id) for prefix_id in plan.prefix_ids]
        token_lengths = [int(plan.token_indptr[row + 1] - plan.token_indptr[row]) for row in range(plan.sequence_count)]
        for prefix_id, token_count in zip(plan.prefix_ids, token_lengths):
            self.kv_manager.reserve_append(prefix_id, token_count)
        page_ids = [self.kv_manager.page_ids(prefix_id) for prefix_id in plan.prefix_ids]
        page_indptr = [0]
        for pages in page_ids:
            page_indptr.append(page_indptr[-1] + len(pages))
        device = self.backend.device
        append_batch_indices = torch.cat([
            torch.full((length,), row, dtype=torch.int32, device=device)
            for row, length in enumerate(token_lengths)
        ])
        append_positions = torch.cat([
            torch.arange(old, old + length, dtype=torch.int32, device=device)
            for old, length in zip(old_lens, token_lengths)
        ])
        batch = PagedBatch(
            plan.token_data.to(device, non_blocking=True),
            plan.token_indptr.to(device, non_blocking=True),
            torch.tensor(page_indptr, device=device, dtype=torch.int32),
            torch.tensor([page for pages in page_ids for page in pages], device=device, dtype=torch.int32),
            torch.tensor(
                [((old + added - 1) % self.kv_store.page_size) + 1 for old, added in zip(old_lens, token_lengths)],
                device=device,
                dtype=torch.int32,
            ),
            torch.tensor(old_lens, device=device, dtype=torch.int32),
            append_batch_indices,
            append_positions,
        )
        hidden = self.backend.run(batch, self.kv_store)
        representations = tuple(None if plan.stage == Stage.STATE else value for value in hidden)
        return BatchResult(
            plan.stage,
            plan.node_ids,
            representations,
            tuple(old + added for old, added in zip(old_lens, token_lengths)),
        )
