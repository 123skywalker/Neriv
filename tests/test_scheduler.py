from jev_like.engine.core import Stage
from jev_like.engine.core.scheduler import Scheduler
from jev_like.engine.core.tables import CandidateRecord, NodeStatus, QuestionRecord, RuntimeTables, StateRecord


def test_scheduler_prefers_candidate_and_builds_ragged_plan() -> None:
    tables = RuntimeTables()
    state_begin, state_len = tables.tokens.append([1, 2, 3])
    state_id = tables.add_state(StateRecord(0, state_begin, state_len))
    question_begin, question_len = tables.tokens.append([4, 5])
    question_id = tables.add_question(QuestionRecord(0, state_id, question_begin, question_len, prefix_id=7))
    candidate_begin, candidate_len = tables.tokens.append([6])
    candidate_id = tables.add_candidate(CandidateRecord(question_id, candidate_begin, candidate_len, prefix_id=7))
    scheduler = Scheduler(tables)
    scheduler.enqueue_state(state_id)
    scheduler.enqueue_candidate(candidate_id)
    plan = scheduler.schedule(8)
    assert plan.stage == Stage.CANDIDATE
    assert plan.node_ids == (candidate_id,)
    assert plan.prefix_ids == (7,)
    assert plan.token_indptr.tolist() == [0, 1]


def test_scheduler_chunks_long_state() -> None:
    tables = RuntimeTables()
    begin, length = tables.tokens.append(list(range(10)))
    state_id = tables.add_state(StateRecord(0, begin, length))
    scheduler = Scheduler(tables)
    scheduler.enqueue_state(state_id)
    plan = scheduler.schedule(4)
    assert plan.token_data.tolist() == [0, 1, 2, 3]


def test_question_uses_isolated_fork_prefix() -> None:
    tables = RuntimeTables()
    state_begin, state_len = tables.tokens.append([1])
    state_id = tables.add_state(StateRecord(0, state_begin, state_len, prefix_id=3))
    begin, length = tables.tokens.append([2])
    question_id = tables.add_question(QuestionRecord(0, state_id, begin, length, prefix_id=9))
    scheduler = Scheduler(tables)
    scheduler.enqueue_question(question_id)
    assert scheduler.schedule(4).prefix_ids == (9,)


def test_cancelled_stale_nodes_do_not_emit_empty_gpu_batch() -> None:
    tables = RuntimeTables()
    begin, length = tables.tokens.append([1])
    state_id = tables.add_state(StateRecord(0, begin, length))
    scheduler = Scheduler(tables)
    scheduler.enqueue_state(state_id)
    tables.states[state_id].status = NodeStatus.CANCELLED
    assert scheduler.schedule(4) is None
