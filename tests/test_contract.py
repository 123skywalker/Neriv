import pytest

from jev_like.model.contract import CandidateInput, DecisionInput, ModelContract, QuestionInput
from jev_like.training.dataset import DecisionCollator, DecisionExample


class _Tokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))


def _input() -> DecisionInput:
    return DecisionInput(
        "r1", "s1", "t1", "t1", "状态",
        (QuestionInput("q1", "选择", (
            CandidateInput("a", "甲"), CandidateInput("b", "乙"),
        )),),
    )


def test_training_and_serving_use_exact_same_token_segments() -> None:
    contract = ModelContract(_Tokenizer(), "qwen", "revision", "revision", 512)
    compiled = contract.compile(_input())
    collator = DecisionCollator(contract, 512, permutation=False, subset_ratio=0.0)
    batch = collator([DecisionExample("状态", "选择", ["a", "b"], ["甲", "乙"], [1.0, 0.0], "choice")])
    query = list(compiled.state_tokens + compiled.questions[0].question_tokens)
    first = query + list(compiled.questions[0].candidate_tokens[0])
    assert batch["query_input_ids"][0, :len(query)].tolist() == query
    assert batch["candidate_input_ids"][0, 0, :len(first)].tolist() == first


def test_contract_rejects_reserved_literal_and_oversized_path() -> None:
    contract = ModelContract(_Tokenizer(), "qwen", "revision", "revision", 64)
    with pytest.raises(ValueError, match="special-token"):
        contract.compile(DecisionInput(
            "r", "s", "t", "t", "<|im_start|>", _input().questions,
        ))
    with pytest.raises(ValueError, match="超过"):
        contract.compile(_input())
