from __future__ import annotations

import time
from collections import deque

import torch

from .batch_plan import BatchPlan, Stage
from .tables import NodeStatus, RuntimeTables


class Scheduler:
    """单所有者、按 token budget 调度整数 ID 的 continuous DAG scheduler。"""

    def __init__(self, tables: RuntimeTables, aging_rate: float = 0.05) -> None:
        self.tables = tables
        self.aging_rate = aging_rate
        self._ready: dict[Stage, deque[int]] = {stage: deque() for stage in Stage}

    def enqueue_state(self, state_id: int) -> None:
        self._enqueue(Stage.STATE, state_id)

    def enqueue_question(self, question_id: int) -> None:
        self._enqueue(Stage.QUESTION, question_id)

    def enqueue_candidate(self, candidate_id: int) -> None:
        self._enqueue(Stage.CANDIDATE, candidate_id)

    def _record(self, stage: Stage, node_id: int):
        if stage == Stage.STATE:
            return self.tables.states[node_id]
        if stage == Stage.QUESTION:
            return self.tables.questions[node_id]
        return self.tables.candidates[node_id]

    def _prefix_id(self, stage: Stage, node_id: int) -> int:
        record = self._record(stage, node_id)
        if stage == Stage.STATE:
            return record.prefix_id
        return record.prefix_id

    def _enqueue(self, stage: Stage, node_id: int) -> None:
        record = self._record(stage, node_id)
        record.status = NodeStatus.READY
        record.enqueued_at = time.monotonic()
        self._ready[stage].append(node_id)

    def _stage(self) -> Stage | None:
        for stage, queue in self._ready.items():
            self._ready[stage] = deque(node_id for node_id in queue
                                       if self._record(stage, node_id).status == NodeStatus.READY)
        available = [stage for stage, queue in self._ready.items() if queue]
        if not available:
            return None
        now = time.monotonic()
        for stage in available:
            ordered = sorted(
                self._ready[stage],
                key=lambda node_id: (
                    self._record(stage, node_id).deadline_ns or 2**63 - 1,
                    -self._record(stage, node_id).priority,
                    self._record(stage, node_id).enqueued_at,
                ),
            )
            self._ready[stage] = deque(ordered)
        return min(
            available,
            key=lambda stage: (
                self._record(stage, self._ready[stage][0]).deadline_ns or 2**63 - 1,
                -self._record(stage, self._ready[stage][0]).priority,
                -float(stage) - self.aging_rate * (now - self._record(stage, self._ready[stage][0]).enqueued_at),
            ),
        )

    def schedule(self, token_budget: int) -> BatchPlan | None:
        """选择单一 stage 的 ragged batch；长 State 会按预算分块。"""

        if token_budget <= 0:
            raise ValueError("token_budget 必须大于 0")
        stage = self._stage()
        if stage is None:
            return None
        queue = self._ready[stage]
        node_ids: list[int] = []
        prefix_ids: list[int] = []
        tokens: list[int] = []
        indptr = [0]
        remaining = token_budget
        while queue and remaining > 0:
            node_id = queue[0]
            record = self._record(stage, node_id)
            if record.status != NodeStatus.READY:
                queue.popleft()
                continue
            offset = record.token_progress if stage == Stage.STATE else 0
            available = record.token_len - offset
            if available > remaining and node_ids:
                break
            take = min(available, remaining)
            queue.popleft()
            piece = self.tables.tokens.slice(record.token_begin, record.token_len, offset, take)
            tokens.extend(piece)
            indptr.append(len(tokens))
            node_ids.append(node_id)
            prefix_ids.append(self._prefix_id(stage, node_id))
            record.status = NodeStatus.RUNNING
            remaining -= take
        if not node_ids:
            return None
        return BatchPlan(
            stage,
            tuple(node_ids),
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(indptr, dtype=torch.int32),
            tuple(prefix_ids),
        )

    def counts(self) -> dict[str, int]:
        return {f"ready_{stage.name.lower()}_count": len(queue) for stage, queue in self._ready.items()}

    def has_ready(self) -> bool:
        return any(self._ready.values())
