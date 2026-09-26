from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from jev_like.data.io import read_jsonl
from jev_like.model.contract import ModelContract


@dataclass(slots=True)
class DecisionExample:
    """从多问题 Canonical Sample 展开的单问题训练样本。

    @param sample_id 原始 Canonical Sample 标识，旧位置参数可省略。
    @param question_id 原始 Question 标识，旧位置参数可省略。
    @param dataset 原始数据集标签，旧位置参数可省略。
    @param candidate_policy 候选构造策略，固定候选不参与 subset。
    @param category Hardening 混合采样类别。
    @param hard_candidate_ids 冻结 Hard Pool 选中的候选标识。
    """

    state: str
    prompt: str
    candidate_ids: list[str]
    candidates: list[str]
    probabilities: list[float]
    question_type: str
    split: str = "train"
    sample_id: str = ""
    question_id: str = ""
    dataset: str = ""
    candidate_policy: str = ""
    category: str = ""
    hard_candidate_ids: list[str] = field(default_factory=list)


class JsonlDecisionDataset(Dataset[DecisionExample]):
    """只读取 Canonical IR，不感知任何 Prompt 模板。"""

    def __init__(self, path: str | Path) -> None:
        self.examples = [
            DecisionExample(
                sample.state,
                question.prompt,
                [candidate.candidate_id for candidate in question.candidates],
                [candidate.text for candidate in question.candidates],
                question.target.probabilities,
                question.type,
                str(sample.meta.get("split", "")),
                sample.sample_id,
                question.question_id,
                str(sample.meta.get("dataset", "")),
                str(sample.meta.get("candidate_policy", "")),
                str(sample.meta.get("category", "")),
                list(sample.meta.get("hard_candidate_ids", [])),
            )
            for sample in read_jsonl(path)
            for question in sample.questions
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DecisionExample:
        return self.examples[index]


class DecisionCollator:
    """通过 Model Contract 编译并执行概率质量守恒的数据增强。

    @param outlier_states 仅供 Reject Refresh 替换 State 的 OOS 训练文本。
    @param permute_score 是否排列 Score 展示顺序；RPS 使用 ordinal_order 复原。
    """

    def __init__(
        self,
        contract: ModelContract,
        max_length: int = 512,
        permutation: bool = True,
        subset_ratio: float = 0.25,
        reject_ratio: float = 0.0,
        mismatch_ratio: float = 0.0,
        seed: int = 42,
        outlier_states: list[str] | None = None,
        outlier_ratio: float = 0.0,
        permute_score: bool = False,
    ) -> None:
        self.contract = contract
        self.max_length = max_length
        self.permutation = permutation
        self.subset_ratio = subset_ratio
        self.reject_ratio = reject_ratio
        self.mismatch_ratio = mismatch_ratio
        self.outlier_states = outlier_states or []
        self.outlier_ratio = outlier_ratio
        self.permute_score = permute_score
        self.random = random.Random(seed)

    def get_random_state(self) -> tuple[Any, ...]:
        """返回候选增强的可恢复随机状态。"""

        return self.random.getstate()

    def set_random_state(self, state: Any) -> None:
        """恢复 JSON 反序列化后的随机状态。"""

        def as_tuple(value: Any) -> Any:
            return tuple(as_tuple(item) for item in value) if isinstance(value, list) else value

        self.random.setstate(as_tuple(state))

    @staticmethod
    def _pad(sequences: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(sequence) for sequence in sequences)
        ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            ids[row, :len(sequence)] = torch.tensor(sequence)
            mask[row, :len(sequence)] = 1
        return ids, mask

    def _prepare(self, example: DecisionExample, force_reject: bool) -> tuple[list[str], list[float], float, list[int]]:
        """同步变换候选、目标概率与原始 ordinal 位置。"""

        pairs = [(candidate, probability, ordinal) for ordinal, (candidate, probability) in
                 enumerate(zip(example.candidates, example.probabilities))]
        if self.permutation and (example.question_type != "score" or self.permute_score):
            self.random.shuffle(pairs)
        if (example.question_type not in {"score", "noul"} and
                example.candidate_policy not in {"nested-k-v1", "fixed-candidates-v1"} and len(pairs) > 2 and
                self.subset_ratio > 0 and self.random.random() < self.subset_ratio):
            remove_count = self.random.randint(1, len(pairs) - 2)
            for index in sorted(self.random.sample(range(len(pairs)), remove_count), reverse=True):
                pairs.pop(index)
        if force_reject and example.question_type not in {"score", "noul"} and len(pairs) >= 3:
            winner = max(range(len(pairs)), key=lambda index: pairs[index][1])
            pairs.pop(winner)
        candidates = [candidate for candidate, _, _ in pairs]
        probabilities = [probability for _, probability, _ in pairs]
        reject = max(0.0, 1.0 - sum(probabilities))
        return candidates, probabilities, reject, [ordinal for _, _, ordinal in pairs]

    def __call__(self, examples: list[DecisionExample]) -> dict[str, torch.Tensor]:
        prepared: list[tuple[DecisionExample, list[str], list[float], float, str, list[int]]] = []
        for row, example in enumerate(examples):
            explicit_reject = self.reject_ratio > 0 and self.random.random() < self.reject_ratio
            candidates, probabilities, reject, ordinals = self._prepare(example, explicit_reject)
            state = example.state
            if len(examples) > 1 and self.mismatch_ratio > 0 and self.random.random() < self.mismatch_ratio:
                state = examples[(row + 1) % len(examples)].state
                probabilities = [0.0] * len(probabilities)
                reject = 1.0
            if self.outlier_states and self.outlier_ratio > 0 and self.random.random() < self.outlier_ratio:
                state = self.random.choice(self.outlier_states)
                probabilities = [0.0] * len(probabilities)
                reject = 1.0
            prepared.append((example, candidates, probabilities, reject, state, ordinals))

        max_candidates = max(len(item[1]) for item in prepared)
        candidate_mask = torch.zeros((len(prepared), max_candidates), dtype=torch.bool)
        targets = torch.zeros((len(prepared), max_candidates + 1), dtype=torch.float32)
        ordinal_mask = torch.zeros(len(prepared), dtype=torch.bool)
        ordinal_order = torch.arange(max_candidates).repeat(len(prepared), 1)
        query_sequences: list[list[int]] = []
        candidate_sequences: list[list[int]] = []
        for row, (example, candidates, probabilities, reject, state, ordinals) in enumerate(prepared):
            state_tokens = list(self.contract.state_tokens(state))
            question_tokens = list(self.contract.question_tokens(example.prompt))
            query = state_tokens + question_tokens
            if len(query) > self.max_length:
                raise ValueError(f"训练 Query token 序列超过 {self.max_length}")
            query_sequences.append(query)
            candidate_mask[row, :len(candidates)] = True
            targets[row, :len(probabilities)] = torch.tensor(probabilities)
            targets[row, -1] = reject
            ordinal_mask[row] = example.question_type == "score" and reject == 0
            if example.question_type == "score":
                ordinal_order[row, :len(ordinals)] = torch.tensor(ordinals).argsort()
            for column in range(max_candidates):
                text = candidates[column] if column < len(candidates) else "padding candidate"
                append = list(self.contract.candidate_tokens(text))
                sequence = state_tokens + question_tokens + append
                if len(sequence) > self.max_length:
                    raise ValueError(f"训练 Candidate token 序列超过 {self.max_length}")
                candidate_sequences.append(sequence)

        pad = self.contract.tokenizer.pad_token_id
        query_ids, query_mask = self._pad(query_sequences, pad)
        candidate_ids, candidate_attention = self._pad(candidate_sequences, pad)
        candidate_length = candidate_ids.shape[1]
        return {
            "query_input_ids": query_ids,
            "query_attention_mask": query_mask,
            "candidate_input_ids": candidate_ids.reshape(len(prepared), max_candidates, candidate_length),
            "candidate_attention_mask": candidate_attention.reshape(
                len(prepared), max_candidates, candidate_length
            ),
            "candidate_mask": candidate_mask,
            "targets": targets,
            "ordinal_mask": ordinal_mask,
            "ordinal_order": ordinal_order,
        }
