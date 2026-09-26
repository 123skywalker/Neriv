from __future__ import annotations

import torch
from torch.nn import functional as F

from jev_like.engine.core.tables import RuntimeTables
from jev_like.engine.storage import ReprStore
from jev_like.model.set_pointer import SetPointerHead


class PointerRunner:
    """按候选数 bucket 的小型 Set Head 执行器。"""

    def __init__(self, pointer: SetPointerHead, repr_store: ReprStore, tables: RuntimeTables,
                 temperature: torch.Tensor) -> None:
        self.pointer = pointer
        self.repr_store = repr_store
        self.tables = tables
        self.temperature = temperature

    @staticmethod
    def _bucket_size(count: int) -> int:
        if count <= 2:
            return 2
        if count <= 4:
            return 4
        if count <= 8:
            return 8
        return 16 if count <= 16 else count

    @torch.inference_mode()
    def score(self, question_ids: list[int]) -> dict[int, torch.Tensor]:
        buckets: dict[int, list[int]] = {}
        for question_id in question_ids:
            count = self.tables.questions[question_id].candidate_count
            buckets.setdefault(self._bucket_size(count), []).append(question_id)
        results: dict[int, torch.Tensor] = {}
        for width, ids in buckets.items():
            query_slots = [self.tables.questions[question_id].query_repr_slot for question_id in ids]
            queries = self.repr_store.read_queries(query_slots)
            candidates = torch.zeros((len(ids), width, queries.shape[-1]), device=queries.device, dtype=queries.dtype)
            mask = torch.zeros((len(ids), width), device=queries.device, dtype=torch.bool)
            for row, question_id in enumerate(ids):
                question = self.tables.questions[question_id]
                records = self.tables.candidates[
                    question.candidate_begin:question.candidate_begin + question.candidate_count
                ]
                slots = [record.repr_slot for record in records]
                values = self.repr_store.read_candidates(slots)
                candidates[row, :len(slots)] = values
                mask[row, :len(slots)] = True
            logits = self.pointer(queries, candidates, mask, self.temperature)
            probabilities = F.softmax(logits.float(), dim=-1)
            for row, question_id in enumerate(ids):
                count = self.tables.questions[question_id].candidate_count
                results[question_id] = torch.cat((probabilities[row, :count], probabilities[row, -1:])).cpu()
        return results
