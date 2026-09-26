from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


EngineStatus = Literal["OK", "OVERLOADED", "DEADLINE_EXCEEDED", "MODEL_CONTRACT_MISMATCH", "CANCELLED", "INPUT_TOO_LARGE", "INTERNAL_ERROR"]


@dataclass(frozen=True, slots=True)
class SubmitReceipt:
    """非阻塞提交的接收结果。"""

    accepted: bool
    request_id: str
    status: EngineStatus


@dataclass(frozen=True, slots=True)
class CandidateResult:
    """普通候选的校准概率。"""

    candidate_id: str
    probability: float


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Engine 对单个 Question 的结构化结果。"""

    request_id: str
    snapshot_id: str
    question_id: str
    candidates: tuple[CandidateResult, ...]
    reject_probability: float
    max_probability: float
    margin: float
    normalized_entropy: float
    selected_candidate_id: str | None
    status: EngineStatus = "OK"


@dataclass(frozen=True, slots=True)
class DecisionResponse:
    """Engine 请求级响应。"""

    request_id: str
    snapshot_id: str
    results: tuple[DecisionResult, ...]
    status: EngineStatus = "OK"
