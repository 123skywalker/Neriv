from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any


MODEL_CONTRACT_VERSION = "neriv-contract-v1"
TEMPLATE_VERSION = "neriv-contract-v1"


@dataclass(frozen=True, slots=True)
class CandidateInput:
    """字符串 Adapter 的候选输入。"""

    candidate_id: str
    text: str


@dataclass(frozen=True, slots=True)
class QuestionInput:
    """字符串 Adapter 的问题输入。"""

    question_id: str
    prompt: str
    candidates: tuple[CandidateInput, ...]


@dataclass(frozen=True, slots=True)
class DecisionInput:
    """进入 Model Contract Compiler 前的字符串请求。"""

    request_id: str
    snapshot_id: str
    tenant_id: str
    cache_scope_id: str
    state: str
    questions: tuple[QuestionInput, ...]
    deadline_ns: int = 0
    priority: int = 0
    actor_id: str = "anonymous"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecisionInput":
        """解析 REST/CLI 字典，不在此处拼接模型 Prompt。"""

        questions = tuple(
            QuestionInput(
                raw["question_id"],
                raw["prompt"],
                tuple(CandidateInput(**candidate) for candidate in raw["candidates"]),
            )
            for raw in value["questions"]
        )
        return cls(
            request_id=str(value["request_id"]),
            snapshot_id=str(value.get("snapshot_id", value["request_id"])),
            tenant_id=str(value.get("tenant_id", "default")),
            cache_scope_id=str(value.get("cache_scope_id", value.get("tenant_id", "default"))),
            state=value["state"],
            questions=questions,
            deadline_ns=int(value.get("deadline_ns", 0)),
            priority=int(value.get("priority", 0)),
            actor_id=str(value.get("actor_id", "anonymous")),
        )


@dataclass(frozen=True, slots=True)
class CompiledQuestion:
    """Engine 热路径问题，仅包含 ID 与 token。"""

    question_id: str
    question_tokens: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    candidate_tokens: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class CompiledDecisionRequest:
    """Engine Core 唯一接受的请求协议。"""

    request_id: str
    snapshot_id: str
    tenant_id: str
    cache_scope_id: str
    state_tokens: tuple[int, ...]
    questions: tuple[CompiledQuestion, ...]
    deadline_ns: int
    priority: int
    model_contract_version: str
    contract_hash: str


class ModelContract:
    """Canonical Serialization 与 tokenize 的唯一实现。"""

    def __init__(
        self,
        tokenizer: Any,
        backbone_id: str,
        backbone_revision: str,
        tokenizer_revision: str,
        max_sequence_tokens: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.backbone_id = backbone_id
        self.backbone_revision = backbone_revision
        self.tokenizer_revision = tokenizer_revision
        self.max_sequence_tokens = max_sequence_tokens
        payload = {
            "version": MODEL_CONTRACT_VERSION,
            "template": TEMPLATE_VERSION,
            "backbone": backbone_id,
            "backbone_revision": backbone_revision,
            "tokenizer_revision": tokenizer_revision,
            "max_sequence_tokens": max_sequence_tokens,
        }
        self.contract_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _text(value: str) -> str:
        normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.strip():
            raise ValueError("Contract 输入文本不能为空")
        if "<|" in normalized or "|>" in normalized:
            raise ValueError("Contract 输入不得包含协议 special-token literal")
        return normalized

    def serialize_state(self, state: str) -> str:
        """序列化共享 State Prefix。"""

        return (
            "<|im_start|>system\nYou are Neriv, a structured decision model.\n<|im_end|>\n"
            f"<|im_start|>user\nState:\n{self._text(state)}"
        )

    def serialize_question(self, prompt: str) -> str:
        """序列化隔离的 Question Append。"""

        return f"\n\nQuestion:\n{self._text(prompt)}\nDecision:"

    def serialize_candidate(self, candidate: str) -> str:
        """序列化独立 Candidate Append。"""

        return f"\nCandidate:\n{self._text(candidate)}\n<|im_end|>"

    def _tokens(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def state_tokens(self, state: str) -> tuple[int, ...]:
        """编译 State Prefix token。"""

        return self._tokens(self.serialize_state(state))

    def question_tokens(self, prompt: str) -> tuple[int, ...]:
        """编译 Question Append token。"""

        return self._tokens(self.serialize_question(prompt))

    def candidate_tokens(self, text: str) -> tuple[int, ...]:
        """编译 Candidate Append token。"""

        return self._tokens(self.serialize_candidate(text))

    def compile_tokens(
        self,
        request_id: str,
        snapshot_id: str,
        tenant_id: str,
        cache_scope_id: str,
        state_tokens: tuple[int, ...],
        questions: tuple[CompiledQuestion, ...],
        deadline_ns: int = 0,
        priority: int = 0,
    ) -> CompiledDecisionRequest:
        """校验预编译 token 并构造 Engine 热路径协议。"""

        if not request_id or not snapshot_id or not tenant_id or not cache_scope_id.strip():
            raise ValueError("请求 ID、租户与 cache_scope_id 不能为空")
        question_ids = [question.question_id for question in questions]
        if not question_ids or len(question_ids) != len(set(question_ids)):
            raise ValueError("question_id 不能为空且必须唯一")
        for question in questions:
            if not 2 <= len(question.candidate_ids) <= 16:
                raise ValueError(f"{question.question_id}: candidates 必须位于 [2,16]")
            if len(question.candidate_ids) != len(set(question.candidate_ids)):
                raise ValueError(f"{question.question_id}: candidate_id 必须唯一")
            if len(question.candidate_ids) != len(question.candidate_tokens):
                raise ValueError(f"{question.question_id}: candidate ID/token 数量不一致")
            longest = max(len(tokens) for tokens in question.candidate_tokens)
            if len(state_tokens) + len(question.question_tokens) + longest > self.max_sequence_tokens:
                raise ValueError(f"{question.question_id}: token 序列超过 {self.max_sequence_tokens}")
        return CompiledDecisionRequest(
            request_id, snapshot_id, tenant_id, cache_scope_id, state_tokens, questions,
            deadline_ns, priority, MODEL_CONTRACT_VERSION, self.contract_hash,
        )

    def compile(self, request: DecisionInput) -> CompiledDecisionRequest:
        """验证字符串请求并编译为 Engine Token Protocol。"""

        if not request.request_id or not request.snapshot_id or not request.tenant_id:
            raise ValueError("request_id、snapshot_id 和 tenant_id 不能为空")
        question_ids = [question.question_id for question in request.questions]
        if not question_ids or len(question_ids) != len(set(question_ids)):
            raise ValueError("question_id 不能为空且必须唯一")
        compiled: list[CompiledQuestion] = []
        for question in request.questions:
            if not 2 <= len(question.candidates) <= 16:
                raise ValueError(f"{question.question_id}: candidates 必须位于 [2,16]")
            candidate_ids = tuple(candidate.candidate_id for candidate in question.candidates)
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"{question.question_id}: candidate_id 必须唯一")
            compiled.append(
                CompiledQuestion(
                    question.question_id,
                    self.question_tokens(question.prompt),
                    candidate_ids,
                    tuple(self.candidate_tokens(candidate.text) for candidate in question.candidates),
                )
            )
        return self.compile_tokens(
            request.request_id, request.snapshot_id, request.tenant_id, request.cache_scope_id,
            self.state_tokens(request.state), tuple(compiled), request.deadline_ns, request.priority,
        )

    def manifest(self) -> dict[str, str]:
        """返回 checkpoint/run 必须固化的 Contract 元数据。"""

        return {
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "contract_hash": self.contract_hash,
            "template_version": TEMPLATE_VERSION,
            "backbone_id": self.backbone_id,
            "backbone_revision": self.backbone_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "max_sequence_tokens": str(self.max_sequence_tokens),
        }
