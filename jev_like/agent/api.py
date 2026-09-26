from pathlib import Path
from typing import Any

from .service import AgentRequest, AgentService, AuthenticationError


def create_app(service: AgentService, mount_ui: bool = True) -> Any:
    """创建统一调用 AgentService 的 FastAPI 与可选 `/ui` 控制台。"""

    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as error:
        raise RuntimeError("API 需要安装 FastAPI") from error

    app = FastAPI(title="Neriv Agent API", version="1.0")

    def token(request: Request) -> str | None:
        """从 Authorization Bearer 头读取服务端令牌。"""

        header = request.headers.get("authorization")
        if header is None:
            return None
        scheme, _, credential = header.partition(" ")
        if scheme.lower() != "bearer" or not credential:
            raise HTTPException(status_code=401, detail="INVALID_AUTHORIZATION")
        return credential

    def call(function, *args):
        try:
            return function(*args)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except AuthenticationError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except TimeoutError as error:
            raise HTTPException(status_code=408, detail="DEADLINE_EXCEEDED") from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/decide")
    def decide(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        return call(lambda: service.decide(AgentRequest.from_dict(payload), token(request)))

    @app.post("/api/v1/events")
    def submit_event(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        return call(service.submit_event, payload, token(request))

    @app.post("/api/v1/sessions")
    def create_session(payload: dict[str, Any]) -> dict[str, Any]:
        return call(service.create_session, payload)

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return call(service.get_session, session_id)

    @app.patch("/api/v1/sessions/{session_id}")
    def update_session(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return call(service.update_session, session_id, payload)

    @app.delete("/api/v1/sessions/{session_id}")
    def close_session(session_id: str) -> dict[str, Any]:
        return call(service.close_session, session_id)

    @app.post("/v1/systemone")
    def system_one(payload: dict[str, Any], http_request: Request) -> dict[str, Any]:
        """将 System One 原生题型适配为临时决策，并保留旧版列表响应。"""

        questions = payload.get("questions")
        native = isinstance(questions, dict) and "decision" in questions
        if questions is None:
            questions = [payload]
        elif isinstance(questions, dict):
            if native:
                questions = [dict(item, id=key) for key, item in questions.items()]
            else:
                questions = [dict(item, type=kind) for kind, values in questions.items()
                             for item in (values if isinstance(values, list) else [values])]
        calls = []
        types = {}
        for index, question in enumerate(questions):
            kind = str(question.get("type", "choice")).upper()
            key = str(question.get("id", f"q{index}"))
            candidates = question.get("candidates") or (["yes", "no"] if kind == "NOUL" else [])
            if native:
                criteria = question.get("criteria")
                if kind == "NOUL":
                    criteria = criteria or {}
                    candidates = [{"id": "yes", "text": f"yes: {criteria.get('true', 'Yes')}"},
                                  {"id": "no", "text": f"no: {criteria.get('false', 'No')}"}]
                elif kind == "CHOICE" and isinstance(criteria, dict):
                    candidates = [{"id": label, "text": f"{label}: {description}"}
                                  for label, description in criteria.items()]
                elif kind == "SCORE" and isinstance(criteria, list):
                    candidates = [{"id": str(level), "text": f"{level}: {description}"}
                                  for level, description in enumerate(criteria)]
            calls.append({"decision_id": key, "ad_hoc": {
                "type": kind,
                "instructions": question.get("instructions", question.get("question", "Choose the best candidate")),
                "candidates": candidates,
            }})
            types[key] = kind
        state = payload.get("state", {})
        if isinstance(state, str):
            state = {"text": state}
        elif not isinstance(state, dict):
            raise HTTPException(status_code=422, detail="state 必须是字符串或对象")
        request = AgentRequest.from_dict({
            "request_id": payload.get("request_id"), "session_id": payload.get("session_id"),
            "timeout_ms": payload.get("timeout_ms", 30_000), "execution_mode": "EVALUATE",
            "state": state,
            "decisions": calls,
        })
        response = call(service.decide, request, token(http_request))
        if native:
            answers = {}
            for item in response["decisions"]:
                result = item["result"]
                probabilities = {candidate["candidate_id"]: candidate["probability"]
                                 for candidate in result["candidates"]}
                total = sum(probabilities.values())
                if total <= 0:
                    raise HTTPException(status_code=422, detail="NO_CANDIDATE_PROBABILITY")
                # JevBench 只接受题目标签，故报告候选条件分布；不把 Reject 伪装成某个标签。
                probabilities = {key: value / total for key, value in probabilities.items()}
                key = result["question_id"]
                kind = types[key].lower()
                answer = {"type": kind}
                if kind == "noul":
                    answer["noul"] = probabilities["yes"]
                else:
                    answer["probabilities"] = probabilities
                    if kind == "choice":
                        answer["choice"] = max(probabilities, key=probabilities.get)
                answers[key] = answer
            return {"request_id": response["request_id"], "status": response["status"],
                    "model": "neriv", "answers": answers}
        return {"request_id": response["request_id"], "status": response["status"], "questions": [
            {"id": item["result"]["question_id"], "type": types[item["result"]["question_id"]],
             "probabilities": item["result"]["candidates"],
             "reject_probability": item["result"]["reject_probability"],
             "selected_candidate_id": item["result"]["selected_candidate_id"],
             **({"p_yes": item["result"]["candidates"][0]["probability"]}
                if types[item["result"]["question_id"]] == "NOUL" else {})}
            for item in response["decisions"]
        ]}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"data": [{"id": "neriv", "object": "model", "contract": service.contract.manifest()}]}

    @app.get("/api/v1/plugins")
    def plugins() -> dict[str, Any]:
        return service.list_plugins()

    @app.post("/api/v1/plugins/refresh")
    def refresh_plugins() -> dict[str, Any]:
        return call(service.refresh_mcp)

    @app.get("/api/v1/plugins/{plugin_id}")
    def plugin(plugin_id: str) -> dict[str, Any]:
        return call(service.get_plugin, plugin_id)

    @app.get("/api/v1/status")
    def status() -> dict[str, Any]:
        return service.get_status()

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/actions/{action_id}/approve")
    def approve(action_id: str, request: Request) -> dict[str, Any]:
        return call(service.approve, action_id, token(request))

    @app.post("/api/v1/actions/{action_id}/reject")
    def reject(action_id: str, request: Request) -> dict[str, str]:
        return call(service.reject, action_id, token(request))

    @app.get("/api/v1/actions")
    def actions(request: Request) -> list[dict[str, Any]]:
        return call(service.list_actions, token(request))

    @app.get("/api/v1/audit")
    def audit(request: Request, limit: int = 100) -> list[dict[str, Any]]:
        return call(service.list_audit, token(request), limit)

    if not mount_ui:
        return app
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).resolve().parent / "static"
    if not (static_dir / "index.html").is_file():
        raise RuntimeError("Web Console 未构建，请先运行 `cd web && npm ci && npm run build`")
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")
    return app
