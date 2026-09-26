from __future__ import annotations

from jev_like.model.contract import CompiledDecisionRequest

from .request import RequestContext
from .tables import CandidateRecord, QuestionRecord, RuntimeTables, StateRecord


class RequestCompiler:
    """将已编译 Token Request 写入紧凑 Runtime Tables。"""

    def __init__(self, tables: RuntimeTables) -> None:
        self.tables = tables

    def compile(
        self,
        internal_id: int,
        request: CompiledDecisionRequest,
        deadline: float | None,
    ) -> RequestContext:
        """创建共享 State、隔离 Question 与 Candidate 记录。"""

        state_tokens = list(request.state_tokens)
        state_begin, state_len = self.tables.tokens.append(state_tokens)
        state_id = self.tables.add_state(
            StateRecord(
                internal_id,
                state_begin,
                state_len,
                pending_questions=len(request.questions),
                deadline_ns=request.deadline_ns,
                priority=request.priority,
            )
        )
        question_begin = len(self.tables.questions)
        for question in request.questions:
            token_begin, token_len = self.tables.tokens.append(list(question.question_tokens))
            question_id = len(self.tables.questions)
            candidate_begin = len(self.tables.candidates)
            self.tables.add_question(QuestionRecord(
                request_id=internal_id,
                state_id=state_id,
                token_begin=token_begin,
                token_len=token_len,
                candidate_begin=candidate_begin,
                candidate_count=len(question.candidate_ids),
                pending_candidates=len(question.candidate_ids),
                deadline_ns=request.deadline_ns,
                priority=request.priority,
            ))
            for candidate_tokens in question.candidate_tokens:
                begin, length = self.tables.tokens.append(list(candidate_tokens))
                self.tables.add_candidate(CandidateRecord(
                    question_id,
                    begin,
                    length,
                    deadline_ns=request.deadline_ns,
                    priority=request.priority,
                ))
        return RequestContext(
            internal_id,
            request,
            state_id,
            question_begin,
            len(request.questions),
            len(request.questions),
            deadline,
        )
