from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import IntEnum


class NodeStatus(IntEnum):
    WAITING = 0
    READY = 1
    RUNNING = 2
    DONE = 3
    CANCELLED = 4


class TokenBuffer:
    """所有 token 的追加式连续缓冲区，Record 仅保存 offset/length。"""

    def __init__(self) -> None:
        self._data = array("q")

    def append(self, tokens: list[int]) -> tuple[int, int]:
        begin = len(self._data)
        self._data.extend(tokens)
        return begin, len(tokens)

    def slice(self, begin: int, length: int, offset: int = 0, limit: int | None = None) -> list[int]:
        start = begin + offset
        stop = begin + length if limit is None else min(begin + length, start + limit)
        return self._data[start:stop].tolist()

    def __len__(self) -> int:
        return len(self._data)


@dataclass(slots=True)
class StateRecord:
    request_id: int
    token_begin: int
    token_len: int
    prefix_id: int = -1
    pending_questions: int = 0
    token_progress: int = 0
    status: NodeStatus = NodeStatus.WAITING
    enqueued_at: float = 0.0
    deadline_ns: int = 0
    priority: int = 0


@dataclass(slots=True)
class QuestionRecord:
    request_id: int
    state_id: int
    token_begin: int
    token_len: int
    prefix_id: int = -1
    candidate_begin: int = 0
    candidate_count: int = 0
    query_repr_slot: int = -1
    pending_candidates: int = 0
    status: NodeStatus = NodeStatus.WAITING
    enqueued_at: float = 0.0
    deadline_ns: int = 0
    priority: int = 0


@dataclass(slots=True)
class CandidateRecord:
    question_id: int
    token_begin: int
    token_len: int
    prefix_id: int = -1
    repr_slot: int = -1
    status: NodeStatus = NodeStatus.WAITING
    enqueued_at: float = 0.0
    deadline_ns: int = 0
    priority: int = 0


class RuntimeTables:
    """只按稳定整数 ID 索引的小型运行记录表。"""

    def __init__(self) -> None:
        self.tokens = TokenBuffer()
        self.states: list[StateRecord] = []
        self.questions: list[QuestionRecord] = []
        self.candidates: list[CandidateRecord] = []

    def add_state(self, record: StateRecord) -> int:
        self.states.append(record)
        return len(self.states) - 1

    def add_question(self, record: QuestionRecord) -> int:
        self.questions.append(record)
        return len(self.questions) - 1

    def add_candidate(self, record: CandidateRecord) -> int:
        self.candidates.append(record)
        return len(self.candidates) - 1
