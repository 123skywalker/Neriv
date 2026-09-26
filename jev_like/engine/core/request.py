from __future__ import annotations

import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import IntEnum

from jev_like.engine.request import DecisionResponse, DecisionResult
from jev_like.model.contract import CompiledDecisionRequest


class RequestStatus(IntEnum):
    QUEUED = 0
    RUNNING = 1
    DONE = 2
    CANCELLED = 3
    FAILED = 4


@dataclass(slots=True)
class RequestContext:
    """控制面请求状态；不持有 KV、候选节点或 GPU Tensor。"""

    internal_id: int
    request: CompiledDecisionRequest
    state_id: int
    question_begin: int
    question_count: int
    pending_questions: int
    deadline: float | None
    submitted_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    status: RequestStatus = RequestStatus.QUEUED
    cancelled: bool = False
    future: Future[DecisionResponse] = field(default_factory=Future)
    results: dict[int, DecisionResult] = field(default_factory=dict)
