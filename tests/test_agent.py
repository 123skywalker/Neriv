import json
import re
from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from jev_like.agent import AgentRequest, AgentService, create_app
from jev_like.engine import CandidateResult, DecisionResponse, DecisionResult
from jev_like.model.contract import ModelContract


class _Tokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) + 1 for character in text]


class _Engine:
    def decide(self, request, timeout):
        self.last_request = request
        results = []
        for question in request.questions:
            selected = question.candidate_ids[-1]
            candidates = tuple(CandidateResult(key, 0.05 if key != selected else 0.9)
                               for key in question.candidate_ids)
            results.append(DecisionResult(request.request_id, request.snapshot_id, question.question_id,
                                          candidates, 0.0, 0.9, 0.85, 0.1, selected))
        return DecisionResponse(request.request_id, request.snapshot_id, tuple(results))

    def metrics(self):
        return {"used_pages": 0}


def _service(max_sequence_tokens: int = 512) -> AgentService:
    return AgentService(_Engine(), ModelContract(_Tokenizer(), "qwen", "v1", "v1", max_sequence_tokens),
                        auth_tokens={"operator-key": {"actor_id": "operator", "permissions": ["execute", "approve"]}})


def _request(session_id: str, mode: str) -> AgentRequest:
    return AgentRequest.from_dict({
        "request_id": f"r-{mode}", "session_id": session_id, "actor_id": "operator",
        "decisions": [{"decision_id": "runtime.next_action"}], "execution_mode": mode,
    })


def _projected_state(service: AgentService) -> dict:
    """从假 tokenizer 的可逆 token 读取 Engine 收到的 State。"""

    text = "".join(chr(token - 1) for token in service.engine.last_request.state_tokens)
    return json.loads(text.split("State:\n", 1)[1])


def _ad_hoc(session_id: str, question: str) -> AgentRequest:
    """构造只提交当前题的会话决策。"""

    return AgentRequest.from_dict({
        "session_id": session_id,
        "decisions": [{"decision_id": question, "ad_hoc": {
            "type": "CHOICE", "instructions": question, "candidates": ["体育", "科技"],
        }}],
    })


def test_session_history_is_visible_on_next_turn_only() -> None:
    service = _service()
    session_id = service.create_session({"state": {"notes": "独立背景"}})["session_id"]
    service.decide(_ad_hoc(session_id, "第一问"))
    assert _projected_state(service) == {"notes": "独立背景"}
    assert service.get_session(session_id)["state"]["history_summary"] == [
        {"q": "第一问", "a": "科技", "r": 0.0},
    ]
    service.decide(_ad_hoc(session_id, "第二问"))
    assert _projected_state(service)["history_summary"] == [
        {"q": "第一问", "a": "科技", "r": 0.0},
    ]


def test_overloaded_turn_does_not_enter_session_history() -> None:
    service = _service()
    session_id = service.create_session({"state": {"notes": "独立背景"}})["session_id"]

    class OverloadedEngine(_Engine):
        def decide(self, request, timeout):
            self.last_request = request
            return DecisionResponse(request.request_id, request.snapshot_id, (), "OVERLOADED")

    service.engine = OverloadedEngine()
    assert service.decide(_ad_hoc(session_id, "失败题"))["status"] == "OVERLOADED"
    assert "history_summary" not in service.get_session(session_id)["state"]
    service.engine = _Engine()
    service.decide(_ad_hoc(session_id, "下一题"))
    assert "history_summary" not in _projected_state(service)


def test_session_history_keeps_latest_sixteen() -> None:
    service = _service(4096)
    session_id = service.create_session({"state": {"notes": "背景"}})["session_id"]
    for index in range(20):
        service.decide(_ad_hoc(session_id, f"第{index}问"))
    history = service.get_session(session_id)["state"]["history_summary"]
    assert len(history) == 16
    assert [item["q"] for item in history] == [f"第{index}问" for index in range(4, 20)]


def test_session_history_truncates_question_and_records_reject() -> None:
    service = _service(4096)
    session_id = service.create_session({"state": {"notes": "保留背景"}})["session_id"]

    class RejectEngine(_Engine):
        def decide(self, request, timeout):
            response = super().decide(request, timeout)
            return replace(response, results=tuple(
                replace(result, selected_candidate_id=None, reject_probability=.8)
                for result in response.results
            ))

    service.engine = RejectEngine()
    service.decide(_ad_hoc(session_id, "长" * 180))
    state = service.get_session(session_id)["state"]
    assert state["notes"] == "保留背景"
    assert state["history_summary"] == [{"q": "长" * 100, "a": "reject", "r": .8}]


def test_plugin_does_not_project_session_history_without_declaration() -> None:
    service = _service()
    session_id = service.create_session({"state": {"runtime_metrics": {}, "notes": "独立背景"}})["session_id"]
    service.decide(_ad_hoc(session_id, "第一问"))
    service.decide(_request(session_id, "EVALUATE"))
    assert _projected_state(service) == {
        "runtime_metrics": {}, "observations": None, "environment": None,
    }


def test_plugin_ignores_unprojected_notes_during_history_budget() -> None:
    service = _service()
    session_id = service.create_session({"state": {
        "runtime_metrics": {}, "notes": "<|不参与插件投影|>",
    }})["session_id"]
    assert service.decide(_request(session_id, "EVALUATE"))["status"] == "OK"
    assert "notes" not in _projected_state(service)


def test_ad_hoc_projects_request_state() -> None:
    service = _service()
    service.decide(AgentRequest.from_dict({
        "state": {"notes": "冠军决赛补时绝杀"},
        "decisions": [{"decision_id": "topic", "ad_hoc": {
            "type": "CHOICE", "instructions": "哪一类", "candidates": ["体育", "科技"],
        }}],
    }))
    assert _projected_state(service) == {"notes": "冠军决赛补时绝杀"}


def test_ad_hoc_projects_session_notes() -> None:
    service = _service()
    session = service.create_session({"state": {"notes": "初始背景"}})
    service.update_session(session["session_id"], {"state": {"notes": "冠军决赛补时绝杀"}})
    service.decide(AgentRequest.from_dict({
        "session_id": session["session_id"],
        "decisions": [{"decision_id": "topic", "ad_hoc": {
            "type": "CHOICE", "instructions": "哪一类", "candidates": ["体育", "科技"],
        }}],
    }))
    assert _projected_state(service) == {"notes": "冠军决赛补时绝杀"}


def test_plugin_only_projects_declared_state() -> None:
    service = _service()
    service.decide(AgentRequest.from_dict({
        "state": {"runtime_metrics": {"gpu_memory": .9}, "unrelated": "不应入模"},
        "decisions": [{"decision_id": "runtime.next_action"}],
    }))
    assert _projected_state(service) == {
        "runtime_metrics": {"gpu_memory": .9}, "observations": None, "environment": None,
    }


def test_plugin_and_ad_hoc_project_union() -> None:
    service = _service()
    service.decide(AgentRequest.from_dict({
        "state": {"runtime_metrics": {}, "notes": "冠军决赛补时绝杀"},
        "decisions": [
            {"decision_id": "runtime.next_action"},
            {"decision_id": "topic", "ad_hoc": {
                "type": "CHOICE", "instructions": "哪一类", "candidates": ["体育", "科技"],
            }},
        ],
    }))
    assert _projected_state(service) == {
        "runtime_metrics": {}, "observations": None, "environment": None,
        "notes": "冠军决赛补时绝杀",
    }


def test_system_one_projects_request_notes() -> None:
    service = _service()
    client = TestClient(create_app(service, mount_ui=False))
    response = client.post("/v1/systemone", json={
        "state": {"notes": "冠军决赛补时绝杀"},
        "questions": [{"id": "topic", "type": "choice", "instructions": "哪一类",
                       "candidates": ["体育", "科技"]}],
    })
    assert response.status_code == 200
    assert _projected_state(service) == {"notes": "冠军决赛补时绝杀"}


def test_system_one_typesafe_choice_projects_string_state_and_probabilities() -> None:
    """TypeSafe 请求的字符串背景与 criteria 经适配后仍由 Engine 决策。"""

    service = _service()

    class RejectEngine(_Engine):
        def decide(self, request, timeout):
            response = super().decide(request, timeout)
            return replace(response, results=tuple(replace(
                result, candidates=(CandidateResult("sports", 0.2), CandidateResult("tech", 0.5)),
                reject_probability=0.3, selected_candidate_id="tech",
            ) for result in response.results))

    service.engine = RejectEngine()
    client = TestClient(create_app(service, mount_ui=False))
    response = client.post("/v1/systemone", json={
        "state": "冠军决赛补时绝杀", "model": "jev-latest",
        "questions": {"decision": {"type": "choice", "instructions": "判断新闻类别",
                                   "criteria": {"sports": "体育赛事", "tech": "科技产品"}}},
    })
    assert response.status_code == 200
    assert _projected_state(service) == {"text": "冠军决赛补时绝杀"}
    assert response.json()["status"] == "OK"
    answer = response.json()["answers"]["decision"]
    assert answer["type"] == "choice"
    assert answer["choice"] in answer["probabilities"]
    assert set(answer["probabilities"]) == {"sports", "tech"}
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0)
    assert answer["probabilities"]["sports"] == pytest.approx(0.2 / 0.7)


def test_system_one_typesafe_noul_and_score() -> None:
    """原生二分类和有序等级响应符合 JevBench 的字段形状。"""

    client = TestClient(create_app(_service(), mount_ui=False))
    for question, expected in [
        ({"type": "noul", "instructions": "是否允许？",
          "criteria": {"false": "条件不足", "true": "条件齐全"}}, "noul"),
        ({"type": "score", "instructions": "评估等级",
          "criteria": ["轻微", "中等", "严重"]}, "probabilities"),
    ]:
        response = client.post("/v1/systemone", json={
            "state": "事件背景", "questions": {"decision": question},
        })
        assert response.status_code == 200
        answer = response.json()["answers"]["decision"]
        assert expected in answer
        if expected == "probabilities":
            assert set(answer[expected]) == {"0", "1", "2"}
            assert sum(answer[expected].values()) == pytest.approx(1.0)
        else:
            assert 0.0 <= answer[expected] <= 1.0


def test_execution_modes_and_approval_are_distinct() -> None:
    service = _service()
    session = service.create_session({"tenant_id": "tenant", "state": {"runtime_metrics": {"gpu_memory": .9}}})
    session_id = session["session_id"]
    evaluated = service.decide(_request(session_id, "EVALUATE"))["decisions"][0]
    assert evaluated["execution"] is None and evaluated["proposed_action"] is None
    assert evaluated["policy"] is None
    proposed = service.decide(_request(session_id, "PROPOSE"), "operator-key")["decisions"][0]
    assert proposed["proposed_action"] and "action_id" not in proposed
    executed = service.decide(_request(session_id, "EXECUTE"), "operator-key")["decisions"][0]
    action_id = executed["action_id"]
    with pytest.raises(PermissionError):
        service.approve(action_id)
    assert service.approve(action_id, "operator-key")["status"] == "EXECUTED"


def test_session_versions_and_rest_surface() -> None:
    service = _service()
    client = TestClient(create_app(service, mount_ui=False))
    session = client.post("/api/v1/sessions", json={"tenant_id": "a", "state": {"runtime_metrics": {}}}).json()
    session_id = session["session_id"]
    updated = client.patch(f"/api/v1/sessions/{session_id}", json={"state": {"observations": []}}).json()
    assert updated["state_version"] == session["state_version"] + 1
    response = client.post("/api/v1/decide", json={"session_id": session_id,
        "decisions": [{"decision_id": "runtime.next_action"}]}).json()
    assert response["status"] == "OK"
    assert client.post("/api/v1/decide", json={"decisions": [], "execution_mode": "BAD"}).status_code == 422
    assert client.get("/v1/models").status_code == 200
    system_one = client.post("/v1/systemone", json={"state": {"runtime_metrics": {}},
        "questions": [{"id": "yesno", "type": "noul", "instructions": "Proceed?"}]}).json()
    assert system_one["questions"][0]["type"] == "NOUL"
    assert "p_yes" in system_one["questions"][0]
    assert client.get("/api/v1/plugins").json()["decisions"][0]["id"] == "runtime.next_action"
    assert client.delete(f"/api/v1/sessions/{session_id}").json()["status"] == "CLOSED"
    assert client.get(f"/api/v1/sessions/{session_id}").status_code == 404


def test_client_claimed_permissions_do_not_authorize_actions() -> None:
    service = _service()
    client = TestClient(create_app(service, mount_ui=False))
    session = client.post("/api/v1/sessions", json={"state": {"runtime_metrics": {"gpu_memory": .9}}}).json()
    payload = {"session_id": session["session_id"], "execution_mode": "EXECUTE",
               "actor_id": "operator", "permissions": ["execute", "approve"],
               "decisions": [{"decision_id": "runtime.next_action"}]}
    assert client.post("/api/v1/decide", json=payload).status_code == 403
    response = client.post("/api/v1/decide", json=payload,
                           headers={"Authorization": "Bearer operator-key"})
    assert response.status_code == 200
    action_id = response.json()["decisions"][0]["action_id"]
    claimed = {"actor_id": "operator", "permissions": ["execute", "approve"]}
    assert client.post(f"/api/v1/actions/{action_id}/approve", json=claimed).status_code == 403
    assert client.post(f"/api/v1/actions/{action_id}/reject", json=claimed).status_code == 403
    assert client.post(f"/api/v1/actions/{action_id}/approve", json=claimed,
                       headers={"Authorization": "Bearer wrong-key"}).status_code == 401
    assert client.post(f"/api/v1/actions/{action_id}/approve", json=claimed,
                       headers={"Authorization": "Bearer operator-key"}).json()["status"] == "EXECUTED"


def test_pending_actions_and_audit_require_approve_token() -> None:
    service = _service()
    service._auth_tokens["executor-key"] = {"actor_id": "executor", "permissions": ["execute"]}
    client = TestClient(create_app(service, mount_ui=False))
    session = client.post("/api/v1/sessions", json={"state": {"runtime_metrics": {"gpu_memory": .9}}}).json()
    response = client.post("/api/v1/decide", json={
        "session_id": session["session_id"], "execution_mode": "EXECUTE",
        "decisions": [{"decision_id": "runtime.next_action"}],
    }, headers={"Authorization": "Bearer operator-key"}).json()
    action_id = response["decisions"][0]["action_id"]
    for path in ("/api/v1/actions", "/api/v1/audit?limit=1"):
        assert client.get(path).status_code == 403
        assert client.get(path, headers={"Authorization": "Bearer wrong-key"}).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer executor-key"}).status_code == 403
    headers = {"Authorization": "Bearer operator-key"}
    assert client.get("/api/v1/actions", headers=headers).json() == [{
        "action_id": action_id,
        "capability_id": service.pending_actions[action_id].capability_id,
        "actor_id": "operator",
        "snapshot_id": service.get_session(session["session_id"])["snapshot_id"],
        "session_id": session["session_id"],
        "created_at": service.pending_actions[action_id].created_at,
        "arguments": {},
    }]
    audit = client.get("/api/v1/audit?limit=1", headers=headers).json()
    assert len(audit) == 1 and audit[0]["action_id"] == action_id
    assert client.get("/api/v1/audit?limit=0", headers=headers).status_code == 422
    service.pending_actions[action_id].created_at = 0
    assert client.get("/api/v1/actions", headers=headers).json() == []


def test_web_console_serves_react_build() -> None:
    client = TestClient(create_app(_service(), mount_ui=True))
    page = client.get("/ui/")
    assert page.status_code == 200
    assert "Neriv Console" in page.text
    script = re.search(r'src="(/ui/assets/[^" ]+\.js)"', page.text)
    assert script is not None
    assert client.get(script.group(1)).status_code == 200


def test_audit_is_bounded_and_expired_action_cannot_execute() -> None:
    service = _service()
    for index in range(10_001):
        service.audit.append({"index": index})
    assert len(service.audit) == 10_000
    session = service.create_session({"state": {"runtime_metrics": {"gpu_memory": .9}}})
    action_id = service.decide(_request(session["session_id"], "EXECUTE"),
                               "operator-key")["decisions"][0]["action_id"]
    service.pending_actions[action_id].created_at = 0
    with pytest.raises(ValueError, match="ACTION_EXPIRED"):
        service.approve(action_id, "operator-key")
    assert action_id not in service.pending_actions


def test_cache_scope_is_server_owned_not_client_claimed() -> None:
    service = _service()
    service._auth_tokens["operator-key"]["cache_scope_id"] = "trusted-scope"
    payload = {"cache_scope_id": "forged", "decisions": [{"decision_id": "runtime.next_action"}],
               "state": {"runtime_metrics": {}}}
    service.decide(AgentRequest.from_dict(payload))
    assert service.engine.last_request.cache_scope_id == "default"
    service.decide(AgentRequest.from_dict(payload), "operator-key")
    assert service.engine.last_request.cache_scope_id == "trusted-scope"
