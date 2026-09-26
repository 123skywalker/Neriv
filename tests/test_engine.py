import time
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import torch
import pytest

from jev_like.engine import EngineConfig, NerivEngine
from jev_like.engine.core.engine import EngineRejected
from jev_like.engine.core.batch_plan import Stage
from jev_like.engine.runner.model_runner import BatchResult
from jev_like.model.contract import CandidateInput, DecisionInput, ModelContract, QuestionInput
from jev_like.model.decision_model import NerivDecisionModel
from jev_like.model.set_pointer import SetPointerHead


class _Tokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) % 31 + 1 for character in text]


class _Model(torch.nn.Module):
    def __init__(self, contract: ModelContract) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)
        self.backbone.config = SimpleNamespace(
            hidden_size=4, num_hidden_layers=1, num_key_value_heads=1, head_dim=4,
        )
        self.pointer = torch.nn.Identity()
        self.contract = contract
        self.register_buffer("temperature", torch.tensor(1.0))


def _request(contract: ModelContract, request_id: str):
    return contract.compile(DecisionInput(
        request_id, request_id, "tenant", "tenant", "共享状态",
        (QuestionInput("q1", "选择", (CandidateInput("a", "甲"), CandidateInput("b", "乙"))),),
    ))


def test_submit_validates_contract_deadline_and_token_limits() -> None:
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    engine = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=4), device="cpu")
    request = _request(contract, "first")
    assert engine.submit(replace(request, contract_hash="bad")).status == "MODEL_CONTRACT_MISMATCH"
    assert engine.submit(replace(request, deadline_ns=time.monotonic_ns() - 1)).status == "DEADLINE_EXCEEDED"
    assert engine.submit(replace(request, state_tokens=tuple(range(8193)))).status == "INPUT_TOO_LARGE"
    with pytest.raises(ValueError, match="cache_scope_id"):
        engine.submit(replace(request, cache_scope_id=""))
    engine.shutdown()


def test_submit_is_nonblocking_and_completion_is_polled() -> None:
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    engine = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=4, page_size=1024), device="cpu")
    receipt = engine.submit(_request(contract, "first"))
    assert receipt.accepted and receipt.request_id == "first"
    deadline = time.monotonic() + 3
    responses = []
    while not responses and time.monotonic() < deadline:
        responses = engine.poll_completion()
        time.sleep(0.01)
    assert responses and responses[0].request_id == "first"
    assert responses[0].status == "INTERNAL_ERROR"  # 测试桩没有 Qwen 层。
    engine.shutdown()


def test_async_decision_releases_all_pages() -> None:
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    engine = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=16, page_size=1024), device="cpu")

    def run(plan):
        for prefix_id, begin, end in zip(plan.prefix_ids, plan.token_indptr[:-1], plan.token_indptr[1:]):
            engine.kv.reserve_append(prefix_id, int(end - begin))
        values = tuple(None if plan.stage == Stage.STATE else torch.ones(4) for _ in plan.node_ids)
        return BatchResult(plan.stage, plan.node_ids, values, tuple(engine.kv.seq_len(key) for key in plan.prefix_ids))

    engine.model_runner.run = run
    engine.pointer_runner.score = lambda ids: {key: torch.tensor([.1, .8, .1]) for key in ids}
    response = engine.decide(_request(contract, "ok"), timeout_seconds=3)
    assert response.status == "OK"
    assert response.results[0].selected_candidate_id == "b"
    assert engine.metrics()["used_pages"] == 0
    engine.shutdown()


def test_cancel_during_gpu_work_defers_page_release() -> None:
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    engine = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=16, page_size=1024), device="cpu")
    started, finish = Event(), Event()

    def run(plan):
        for prefix_id, begin, end in zip(plan.prefix_ids, plan.token_indptr[:-1], plan.token_indptr[1:]):
            engine.kv.reserve_append(prefix_id, int(end - begin))
        started.set()
        assert finish.wait(3)
        return BatchResult(plan.stage, plan.node_ids, tuple(None for _ in plan.node_ids), ())

    engine.model_runner.run = run
    assert engine.submit(_request(contract, "cancel-me")).accepted
    assert started.wait(3)
    assert engine.cancel("cancel-me")
    assert engine.kv_store.free_page_count < 16
    finish.set()
    deadline = time.monotonic() + 3
    responses = []
    while not responses and time.monotonic() < deadline:
        responses = engine.poll_completion()
        time.sleep(.01)
    assert responses[0].status == "CANCELLED"
    assert engine.metrics()["used_pages"] == 0
    engine.shutdown()


def test_overload_status_is_typed_not_message_based() -> None:
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    engine = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=1), device="cpu")
    with pytest.raises(EngineRejected) as failure:
        engine._submit(_request(contract, "too-large"))
    assert failure.value.status == "OVERLOADED"
    engine.shutdown()


def test_shutdown_completes_pending_request_before_gpu_returns() -> None:
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    engine = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=16, page_size=1024), device="cpu")
    started, finish = Event(), Event()

    def run(plan):
        started.set()
        finish.wait(3)
        return BatchResult(plan.stage, plan.node_ids, tuple(None for _ in plan.node_ids), ())

    engine.model_runner.run = run
    assert engine.submit(_request(contract, "pending")).accepted
    assert started.wait(3)
    engine.shutdown(timeout_seconds=.02)
    assert not engine._awaiters
    assert [item.status for item in engine.poll_completion()] == ["CANCELLED"]
    finish.set()
    engine._worker.join(timeout=3)
    engine._gpu_worker.join(timeout=3)
    assert engine.poll_completion() == []


def test_cross_request_state_cache_hits_suffix_and_isolates_scope() -> None:
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    engine = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=128, page_size=16), device="cpu")
    state_tokens = contract.state_tokens("A" * 80)
    request = replace(_request(contract, "first"), state_tokens=state_tokens, cache_scope_id="a")
    computed = []

    def run(plan):
        for prefix_id, begin, end in zip(plan.prefix_ids, plan.token_indptr[:-1], plan.token_indptr[1:]):
            engine.kv.reserve_append(prefix_id, int(end - begin))
        if plan.stage == Stage.STATE:
            computed.append(plan.token_count)
        values = tuple(None if plan.stage == Stage.STATE else torch.ones(4) for _ in plan.node_ids)
        return BatchResult(plan.stage, plan.node_ids, values, tuple(engine.kv.seq_len(key) for key in plan.prefix_ids))

    engine.model_runner.run = run
    engine.pointer_runner.score = lambda ids: {key: torch.tensor([.1, .8, .1]) for key in ids}
    assert engine.decide(request, 3).status == "OK"
    first = sum(computed)
    assert first == len(state_tokens)
    computed.clear()
    assert engine.decide(replace(request, request_id="same"), 3).status == "OK"
    assert sum(computed) < first
    computed.clear()
    longer = state_tokens + tuple(range(20))
    assert engine.decide(replace(request, request_id="longer", state_tokens=longer), 3).status == "OK"
    assert 0 < sum(computed) < len(longer)
    computed.clear()
    assert engine.decide(replace(request, request_id="other", cache_scope_id="b"), 3).status == "OK"
    assert sum(computed) == first
    assert engine.metrics()["cached_pages"] > 0
    engine.shutdown()


def test_prefix_cache_can_be_disabled_for_reference_runs() -> None:
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    engine = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=128, page_size=16,
                                                        prefix_cache=False), device="cpu")
    state_lengths = []

    def run(plan):
        for prefix_id, begin, end in zip(plan.prefix_ids, plan.token_indptr[:-1], plan.token_indptr[1:]):
            engine.kv.reserve_append(prefix_id, int(end - begin))
        if plan.stage == Stage.STATE:
            state_lengths.append(plan.token_count)
        values = tuple(None if plan.stage == Stage.STATE else torch.ones(4) for _ in plan.node_ids)
        return BatchResult(plan.stage, plan.node_ids, values, tuple(engine.kv.seq_len(key) for key in plan.prefix_ids))

    engine.model_runner.run = run
    engine.pointer_runner.score = lambda ids: {key: torch.tensor([.1, .8, .1]) for key in ids}
    request = _request(contract, "one")
    assert engine.decide(request, 3).status == "OK"
    assert engine.decide(replace(request, request_id="two"), 3).status == "OK"
    assert state_lengths == [len(request.state_tokens)] * 2
    assert engine.metrics()["cached_pages"] == 0
    engine.shutdown()


def test_cached_pages_are_counted_before_admission() -> None:
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    request = _request(contract, "warm")
    probe = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=128, page_size=16), device="cpu")
    full_pages = probe._required_pages(request)
    probe.shutdown()
    cached_pages = len(request.state_tokens) // 16
    assert cached_pages > 0
    engine = NerivEngine(_Model(contract), EngineConfig(
        max_kv_pages=3 * full_pages - 2 * cached_pages, page_size=16,
    ), device="cpu")
    started, release = Event(), Event()
    block_question = False

    def run(plan):
        nonlocal block_question
        for prefix_id, begin, end in zip(plan.prefix_ids, plan.token_indptr[:-1], plan.token_indptr[1:]):
            engine.kv.reserve_append(prefix_id, int(end - begin))
        if block_question and plan.stage == Stage.QUESTION:
            block_question = False
            started.set()
            assert release.wait(3)
        values = tuple(None if plan.stage == Stage.STATE else torch.ones(4) for _ in plan.node_ids)
        return BatchResult(plan.stage, plan.node_ids, values, tuple(engine.kv.seq_len(key) for key in plan.prefix_ids))

    engine.model_runner.run = run
    engine.pointer_runner.score = lambda ids: {key: torch.tensor([.1, .8, .1]) for key in ids}
    try:
        assert engine.decide(request, 3).status == "OK"
        block_question = True
        assert engine.submit(replace(request, request_id="active")).accepted
        assert started.wait(3)
        assert engine.submit(replace(request, request_id="shared")).accepted
        assert engine.submit(replace(request, request_id="third")).accepted
        deadline = time.monotonic() + 2
        while "third" not in engine._public_ids and time.monotonic() < deadline:
            assert not engine.poll_completion()
            time.sleep(.01)
        assert "third" in engine._public_ids
        assert engine.metrics()["reserved_pages"] == 3 * (full_pages - cached_pages)
        with engine._lock, pytest.raises(EngineRejected) as failure:
            engine._submit(replace(request, request_id="over-capacity"))
        assert failure.value.status == "OVERLOADED"
        release.set()
        statuses = {}
        deadline = time.monotonic() + 3
        while len(statuses) < 3 and time.monotonic() < deadline:
            statuses.update({item.request_id: item.status for item in engine.poll_completion()})
            time.sleep(.01)
        assert statuses == {"active": "OK", "shared": "OK", "third": "OK"}
        assert engine.metrics()["reserved_pages"] == 0
    finally:
        release.set()
        engine.shutdown()


@pytest.mark.parametrize("cancel_first", [False, True])
def test_state_publish_transfers_reservation_to_shared_cache(cancel_first: bool) -> None:
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    request = _request(contract, "first")
    probe = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=128, page_size=16), device="cpu")
    full_pages = probe._required_pages(request)
    probe.shutdown()
    cached_pages = len(request.state_tokens) // 16
    engine = NerivEngine(_Model(contract), EngineConfig(
        max_kv_pages=2 * full_pages - cached_pages, page_size=16,
    ), device="cpu")
    started, release = Event(), Event()

    def run(plan):
        for prefix_id, begin, end in zip(plan.prefix_ids, plan.token_indptr[:-1], plan.token_indptr[1:]):
            engine.kv.reserve_append(prefix_id, int(end - begin))
        if plan.stage == Stage.QUESTION and not started.is_set():
            started.set()
            assert release.wait(3)
        values = tuple(None if plan.stage == Stage.STATE else torch.ones(4) for _ in plan.node_ids)
        return BatchResult(plan.stage, plan.node_ids, values, tuple(engine.kv.seq_len(key) for key in plan.prefix_ids))

    engine.model_runner.run = run
    engine.pointer_runner.score = lambda ids: {key: torch.tensor([.1, .8, .1]) for key in ids}
    try:
        assert engine.submit(request).accepted
        assert started.wait(3)
        assert engine.kv.active_cached_page_count == cached_pages
        assert engine.metrics()["reserved_pages"] == full_pages - cached_pages
        assert engine.submit(replace(request, request_id="shared")).accepted
        deadline = time.monotonic() + 2
        while "shared" not in engine._public_ids and time.monotonic() < deadline:
            assert not engine.poll_completion()
            time.sleep(.01)
        assert "shared" in engine._public_ids
        if cancel_first:
            assert engine.cancel("first")
            while not engine.contexts[engine._public_ids["first"]].cancelled and time.monotonic() < deadline:
                time.sleep(.01)
            assert engine.contexts[engine._public_ids["first"]].cancelled
        release.set()
        statuses = {}
        deadline = time.monotonic() + 3
        while len(statuses) < 2 and time.monotonic() < deadline:
            statuses.update({item.request_id: item.status for item in engine.poll_completion()})
            time.sleep(.01)
        assert statuses == {"first": "CANCELLED" if cancel_first else "OK", "shared": "OK"}
        assert engine.metrics()["reserved_pages"] == 0
    finally:
        release.set()
        engine.shutdown()


def test_concurrent_duplicate_miss_keeps_its_private_state_reservation() -> None:
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    request = _request(contract, "first")
    probe = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=128, page_size=16), device="cpu")
    full_pages = probe._required_pages(request)
    probe.shutdown()
    cached_pages = len(request.state_tokens) // 16
    engine = NerivEngine(_Model(contract), EngineConfig(max_kv_pages=2 * full_pages, page_size=16), device="cpu")
    state_started, release_state = Event(), Event()
    second_question_started, release_question = Event(), Event()

    def run(plan):
        for prefix_id, begin, end in zip(plan.prefix_ids, plan.token_indptr[:-1], plan.token_indptr[1:]):
            engine.kv.reserve_append(prefix_id, int(end - begin))
        if plan.stage == Stage.STATE and not state_started.is_set():
            state_started.set()
            assert release_state.wait(3)
        if plan.stage == Stage.QUESTION and not second_question_started.is_set():
            question = engine.tables.questions[plan.node_ids[0]]
            if engine.contexts[question.request_id].request.request_id == "second":
                second_question_started.set()
                assert release_question.wait(3)
        values = tuple(None if plan.stage == Stage.STATE else torch.ones(4) for _ in plan.node_ids)
        return BatchResult(plan.stage, plan.node_ids, values, tuple(engine.kv.seq_len(key) for key in plan.prefix_ids))

    engine.model_runner.run = run
    engine.pointer_runner.score = lambda ids: {key: torch.tensor([.1, .8, .1]) for key in ids}
    try:
        assert engine.submit(request).accepted
        assert state_started.wait(3)
        assert engine.submit(replace(request, request_id="second", priority=1)).accepted
        deadline = time.monotonic() + 2
        while "second" not in engine._public_ids and time.monotonic() < deadline:
            time.sleep(.01)
        assert "second" in engine._public_ids
        release_state.set()
        assert second_question_started.wait(3)
        assert engine.kv.active_cached_page_count == cached_pages
        assert engine.metrics()["reserved_pages"] == 2 * full_pages - cached_pages
        release_question.set()
        statuses = {}
        deadline = time.monotonic() + 3
        while len(statuses) < 2 and time.monotonic() < deadline:
            statuses.update({item.request_id: item.status for item in engine.poll_completion()})
            time.sleep(.01)
        assert statuses == {"first": "OK", "second": "OK"}
        assert engine.metrics()["reserved_pages"] == 0
    finally:
        release_state.set()
        release_question.set()
        engine.shutdown()


def test_cache_probabilities_match_reference_after_hit_and_eviction() -> None:
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(7)
    contract = ModelContract(_Tokenizer(), "test", "v1", "v1")
    backbone = Qwen3Model(Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=512,
    ))
    model = NerivDecisionModel(backbone, SetPointerHead(32, pointer_size=32), contract)
    request = replace(_request(contract, "baseline"), state_tokens=contract.state_tokens("A" * 64))
    probe = NerivEngine(model, EngineConfig(max_kv_pages=128, page_size=16), device="cpu")
    page_limit = probe._required_pages(request)
    probe.shutdown()

    def probabilities(response):
        assert response.status == "OK"
        result = response.results[0]
        return torch.tensor([*(item.probability for item in result.candidates), result.reject_probability])

    plain = NerivEngine(model, EngineConfig(max_kv_pages=page_limit, page_size=16,
                                            prefix_cache=False), device="cpu")
    try:
        expected = probabilities(plain.decide(request, 10))
    finally:
        plain.shutdown()

    cached = NerivEngine(model, EngineConfig(max_kv_pages=page_limit, page_size=16), device="cpu")
    try:
        for request_id, scope in (("first", "a"), ("hit", "a"), ("evict", "b"), ("recompute", "a")):
            actual = probabilities(cached.decide(replace(request, request_id=request_id,
                                                         cache_scope_id=scope), 10))
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            if request_id == "hit":
                assert cached.metrics()["cache_hits"] == 1
            if request_id == "evict":
                assert cached.metrics()["cache_evictions"] > 0
        metrics = cached.metrics()
        assert metrics["cache_misses"] == 3
    finally:
        cached.shutdown()
