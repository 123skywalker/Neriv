from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Literal
from threading import RLock
from typing import Any

from jev_like.engine import DecisionResponse, NerivEngine
from jev_like.model.contract import CompiledQuestion, ModelContract

from .builtin import register_builtin_plugins
from .plugins import CompiledCapability, CompiledDecision
from .policy import PolicyDecision, PolicyGate, validate_arguments
from .registry import PluginRegistry
from .state import StateSnapshot, StateStore


class AuthenticationError(PermissionError):
    """服务端令牌无效。"""


@dataclass(frozen=True, slots=True)
class DecisionCall:
    """一次显式决策调用及可选临时定义。"""

    decision_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    ad_hoc: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """REST、SDK、MCP 共用的根请求。"""

    request_id: str
    decisions: tuple[DecisionCall, ...]
    session_id: str | None = None
    snapshot_id: str | None = None
    timeout_ms: int = 30_000
    priority: int = 0
    execution_mode: Literal["EVALUATE", "PROPOSE", "EXECUTE"] = "EVALUATE"
    state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentRequest":
        """解析 REST、SDK、MCP 共用的类型化请求。"""

        mode = str(value.get("execution_mode", "EVALUATE")).upper()
        if mode not in {"EVALUATE", "PROPOSE", "EXECUTE"}:
            raise ValueError("未知 execution_mode")
        calls = tuple(
            DecisionCall(str(item["decision_id"]), dict(item.get("parameters", {})), item.get("ad_hoc"))
            for item in value.get("decisions", ())
        )
        return cls(
            str(value.get("request_id") or uuid.uuid4()), calls,
            value.get("session_id"), value.get("snapshot_id"),
            int(value.get("timeout_ms", 30_000)), int(value.get("priority", 0)),
            mode, dict(value.get("state", {})),
        )


@dataclass(slots=True)
class Session:
    """会话保存租户上下文、有界历史和启用的插件，不持有模型内部状态。"""

    session_id: str
    tenant_id: str
    state: StateStore = field(default_factory=StateStore)
    enabled_decisions: tuple[str, ...] = ()
    enabled_capabilities: tuple[str, ...] = ()
    status: str = "ACTIVE"


@dataclass(slots=True)
class PendingAction:
    """等待人工审批的高风险动作。"""

    action_id: str
    snapshot_id: str
    capability_id: str
    arguments: dict[str, Any]
    actor_id: str
    tenant_id: str
    session_id: str | None = None
    created_at: float = field(default_factory=time.time)


class AgentService:
    """REST 与 MCP 共用的单一 Agent Runtime。"""

    def __init__(
        self,
        engine: NerivEngine,
        contract: ModelContract,
        registry: PluginRegistry | None = None,
        state_store: StateStore | None = None,
        mcp_servers: dict[str, str] | None = None,
        auth_tokens: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.engine = engine
        self.contract = contract
        self.registry = registry or PluginRegistry(contract)
        if registry is None:
            register_builtin_plugins(self.registry)
        self._state_stores: dict[str, StateStore] = {}
        if state_store is not None:
            self._state_stores["default"] = state_store
        self.policy = PolicyGate()
        self.sessions: dict[str, Session] = {}
        self.pending_actions: dict[str, PendingAction] = {}
        self._action_states: dict[str, StateStore] = {}
        self.audit: deque[dict[str, Any]] = deque(maxlen=10_000)
        self._auth_tokens = auth_tokens or {}
        self._lock = RLock()
        self._mcp_clients = []
        if mcp_servers:
            from .mcp_client import McpCapabilityClient
            for server_id, url in mcp_servers.items():
                client = McpCapabilityClient(self.registry, server_id, url)
                client.refresh()
                self._mcp_clients.append(client)

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        """创建独立会话并返回首个不可变状态版本。"""

        with self._lock:
            session_id = str(payload.get("session_id") or uuid.uuid4())
            if session_id in self.sessions:
                raise ValueError("重复 session_id")
            session = Session(
                session_id, str(payload.get("tenant_id", "default")),
                enabled_decisions=tuple(payload.get("enabled_decisions", ())),
                enabled_capabilities=tuple(payload.get("enabled_capabilities", ())),
            )
            if payload.get("state"):
                session.state.update(dict(payload["state"]))
            self.sessions[session_id] = session
            return self.get_session(session_id)

    def get_session(self, session_id: str) -> dict[str, Any]:
        """读取会话状态和版本。"""

        session = self.sessions[session_id]
        snapshot = session.state.snapshot()
        return {"session_id": session_id, "tenant_id": session.tenant_id,
                "snapshot_id": snapshot.snapshot_id, "state_version": snapshot.version,
                "state": snapshot.data, "enabled_decisions": session.enabled_decisions,
                "enabled_capabilities": session.enabled_capabilities, "status": session.status}

    def update_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """原子更新状态，每次更新生成新的 snapshot 版本。"""

        with self._lock:
            session = self.sessions[session_id]
            if session.status != "ACTIVE":
                raise RuntimeError("SESSION_CLOSED")
            if "state" in payload:
                session.state.update(dict(payload["state"]))
            if "enabled_decisions" in payload:
                session.enabled_decisions = tuple(payload["enabled_decisions"])
            if "enabled_capabilities" in payload:
                session.enabled_capabilities = tuple(payload["enabled_capabilities"])
            return self.get_session(session_id)

    def close_session(self, session_id: str) -> dict[str, Any]:
        """关闭会话，禁止后续决策和更新。"""

        with self._lock:
            result = self.get_session(session_id)
            del self.sessions[session_id]
            for key, action in tuple(self.pending_actions.items()):
                if action.session_id == session_id:
                    self.pending_actions.pop(key)
                    self._action_states.pop(key, None)
            return {**result, "status": "CLOSED"}

    def resolve_actor(self, token: str | None) -> tuple[str, frozenset[str], str]:
        """仅从服务端令牌表解析身份、权限与缓存隔离域。"""

        if token is None:
            return "anonymous", frozenset(), "default"
        identity = self._auth_tokens.get(token)
        if identity is None:
            raise AuthenticationError("INVALID_TOKEN")
        scope = str(identity.get("cache_scope_id", identity.get("tenant_id", "default")))
        if not scope.strip():
            raise ValueError("cache_scope_id 不能为空")
        return str(identity["actor_id"]), frozenset(identity.get("permissions", ())), scope

    def decide(self, request: AgentRequest, token: str | None = None) -> dict[str, Any]:
        """按同一快照聚合根请求，插件按声明投影，临时题读取完整状态。"""

        if request.timeout_ms <= 0 or not request.decisions:
            raise ValueError("timeout_ms 必须大于 0，decisions 不能为空")
        actor_id, permissions, cache_scope_id = self.resolve_actor(token)
        if request.execution_mode != "EVALUATE" and "execute" not in permissions:
            raise PermissionError("EXECUTION_PERMISSION_DENIED")
        with self._lock:
            self._prune_actions()
            session = self.sessions[request.session_id] if request.session_id else None
            if session and session.status != "ACTIVE":
                raise RuntimeError("SESSION_CLOSED")
            if session and request.state:
                raise ValueError("会话状态必须通过 sessions 更新，不得在 decide 中修改")
            tenant_id = session.tenant_id if session else "default"
            state_store = session.state if session else self._state_stores.get(tenant_id) or StateStore()
            snapshot = state_store.update(request.state) if request.state else state_store.snapshot()
            if request.snapshot_id and request.snapshot_id != snapshot.snapshot_id:
                raise RuntimeError("STALE_SNAPSHOT")
            decision_ids = tuple(dict.fromkeys(call.decision_id for call in request.decisions if call.ad_hoc is None))
            if session and session.enabled_decisions and any(key not in session.enabled_decisions for key in decision_ids):
                raise PermissionError("DECISION_DISABLED")
            decisions = self.registry.decisions_for("MANUAL", decision_ids) if decision_ids else []
            for decision in decisions:
                self.policy.get(decision.descriptor.policy_ref)
            by_id = {item.descriptor.id: item for item in decisions}
            fields = {field for item in decisions for field in item.descriptor.required_state}
            if any(call.ad_hoc is not None for call in request.decisions):
                fields.update(snapshot.data)
            canonical_state = state_store.project(snapshot, fields)
            questions: list[CompiledQuestion] = []
            resolved: dict[str, list[CompiledCapability]] = {}
            history_sources: dict[str, tuple[str, dict[str, str]]] = {}
            seen: set[tuple[str, tuple[str, ...]]] = set()
            for call in request.decisions:
                if call.ad_hoc:
                    spec = call.ad_hoc
                    if spec.get("type") not in {"CHOICE", "SCORE", "NOUL"}:
                        raise ValueError("未知临时 Decision 类型")
                    candidates = spec.get("candidates", ())
                    ids = tuple(str(item.get("id", index)) if isinstance(item, dict) else str(index)
                                for index, item in enumerate(candidates))
                    texts = tuple(str(item.get("text", item.get("id"))) if isinstance(item, dict) else str(item)
                                  for item in candidates)
                    prompt = str(spec["instructions"])
                    question_tokens = self.contract.question_tokens(prompt)
                    candidate_tokens = tuple(self.contract.candidate_tokens(text) for text in texts)
                    resolved[call.decision_id] = []
                else:
                    decision = by_id.get(call.decision_id)
                    if decision is None:
                        raise KeyError(call.decision_id)
                    capabilities = self.registry.resolve_candidates(
                        decision.descriptor.candidate_query, snapshot,
                        None if request.execution_mode == "EVALUATE" else set(permissions),
                    )
                    if session and session.enabled_capabilities:
                        capabilities = [item for item in capabilities if item.descriptor.id in session.enabled_capabilities]
                    ids = tuple(item.descriptor.id for item in capabilities)
                    question_tokens = decision.question_tokens
                    candidate_tokens = tuple(item.candidate_tokens for item in capabilities)
                    resolved[call.decision_id] = capabilities
                if not 2 <= len(ids) <= 16:
                    raise ValueError("候选数量必须位于 [2,16]")
                if session:
                    if call.ad_hoc:
                        history_sources[call.decision_id] = (prompt, dict(zip(ids, texts)))
                    else:
                        history_sources[call.decision_id] = (
                            decision.descriptor.question,
                            {item.descriptor.id: item.descriptor.model_text for item in capabilities},
                        )
                key = (call.decision_id, ids)
                if key not in seen:
                    seen.add(key)
                    questions.append(CompiledQuestion(call.decision_id, question_tokens, ids, candidate_tokens))
            compiled = self.contract.compile_tokens(
                request.request_id, snapshot.snapshot_id, tenant_id, cache_scope_id,
                self.contract.state_tokens(canonical_state), tuple(questions),
                time.monotonic_ns() + request.timeout_ms * 1_000_000, request.priority,
            )
        response = self.engine.decide(compiled, request.timeout_ms / 1000)
        retry_policy = self.policy.get(decisions[0].descriptor.policy_ref) if decisions else None
        if (response.status == "OVERLOADED" and retry_policy and retry_policy.max_retries
                and compiled.deadline_ns - time.monotonic_ns() > retry_policy.retry_backoff_ms * 2_000_000):
            time.sleep(retry_policy.retry_backoff_ms / 1000)
            response = self.engine.decide(compiled, request.timeout_ms / 1000)
        with self._lock:
            outcomes = self._apply_policy(response, decisions, resolved, snapshot, request,
                                          state_store, tenant_id, actor_id, permissions)
            if session and response.status == "OK" and response.results:
                latest = session.state.snapshot()
                stored = latest.data.get("history_summary")
                history = list(stored) if isinstance(stored, list) else []
                appended = False
                for result in response.results:
                    if result.status != "OK":
                        continue
                    prompt, candidates = history_sources[result.question_id]
                    history.append({"q": prompt[:100],
                                    "a": candidates.get(result.selected_candidate_id, "reject"),
                                    "r": result.reject_probability})
                    appended = True
                if appended:
                    updated = session.state.update({"history_summary": self._trim_history(latest, history)})
                    for outcome in outcomes:
                        action_id = outcome.get("action_id")
                        if action_id:
                            self.pending_actions[action_id].snapshot_id = updated.snapshot_id
            if response.status != "OK":
                self.audit.append({"timestamp": time.time(), "actor_id": actor_id,
                                   "request_id": request.request_id, "engine_status": response.status})
            return {"request_id": response.request_id, "snapshot_id": response.snapshot_id,
                    "status": response.status, "decisions": outcomes}

    def _trim_history(self, snapshot: StateSnapshot, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """先截短题干，再按 Contract token 预算从最旧条裁剪历史。"""

        history = history[-16:]
        prefix_tokens = len(self.contract.state_tokens("{}"))
        try:
            notes_tokens = (len(self.contract.state_tokens(StateStore.project(snapshot, {"notes"}))) - prefix_tokens
                            if "notes" in snapshot.data else 0)
        except ValueError:
            return []
        budget = max(0, min(2048, min(4096, self.contract.max_sequence_tokens) - prefix_tokens - notes_tokens - 256))
        while history:
            view = StateSnapshot(snapshot.snapshot_id, snapshot.version, {"history_summary": history})
            try:
                used = len(self.contract.state_tokens(StateStore.project(view, {"history_summary"}))) - prefix_tokens
            except ValueError:
                used = budget + 1
            if used <= budget:
                break
            history.pop(0)
        return history

    def submit_event(self, event: dict[str, Any], token: str | None = None) -> dict[str, Any]:
        """把事件类型映射为 trigger，并复用 decide 完整路径。"""

        payload = dict(event)
        payload["decisions"] = [{"decision_id": item.descriptor.id} for item in self.registry.decisions_for(str(event["type"]))]
        return self.decide(AgentRequest.from_dict(payload), token)

    def _apply_policy(
        self,
        response: DecisionResponse,
        decisions: list[CompiledDecision],
        resolved: dict[str, list[CompiledCapability]],
        snapshot: StateSnapshot,
        request: AgentRequest,
        state_store: StateStore,
        tenant_id: str,
        actor_id: str,
        permissions: frozenset[str],
    ) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        by_id = {decision.descriptor.id: decision for decision in decisions}
        parameters = {call.decision_id: call.parameters for call in request.decisions}
        for result in response.results:
            decision = by_id.get(result.question_id)
            capability_records = {item.descriptor.id: item for item in resolved[result.question_id]}
            capability = capability_records.get(result.selected_candidate_id or "")
            policy = self.policy.check(
                result, decision.descriptor.policy_ref, capability.descriptor if capability else None,
                state_store.is_current(snapshot.snapshot_id), set(permissions), request.execution_mode,
            ) if decision and request.execution_mode != "EVALUATE" else None
            outcome = {"result": asdict(result), "policy": asdict(policy) if policy else None,
                       "proposed_action": None, "execution": None}
            if capability and request.execution_mode != "EVALUATE" and policy.decision in {PolicyDecision.ALLOW, PolicyDecision.REQUIRE_APPROVAL}:
                validate_arguments(capability.descriptor.parameter_schema,
                                   parameters.get(result.question_id, {}))
                if not self.registry.implementation(capability.descriptor.id).available(snapshot.data):
                    raise RuntimeError("EXECUTION_PRECONDITION_CHANGED")
                outcome["proposed_action"] = {"capability_id": capability.descriptor.id,
                                              "arguments": parameters.get(result.question_id, {})}
            if capability is not None and request.execution_mode == "EXECUTE" and policy.decision == PolicyDecision.ALLOW:
                outcome["execution"] = self._execute(
                    capability.descriptor.id,
                    snapshot,
                    parameters.get(result.question_id, {}),
                    state_store,
                )
            elif capability is not None and request.execution_mode == "EXECUTE" and policy.decision == PolicyDecision.REQUIRE_APPROVAL:
                action = PendingAction(
                    str(uuid.uuid4()), snapshot.snapshot_id, capability.descriptor.id,
                    parameters.get(result.question_id, {}), actor_id,
                    tenant_id, request.session_id,
                )
                self.pending_actions[action.action_id] = action
                if request.session_id is None:
                    self._action_states[action.action_id] = state_store
                outcome["action_id"] = action.action_id
            self.audit.append({
                "timestamp": time.time(),
                "actor_id": actor_id,
                "decision_id": result.question_id,
                "probabilities": {item.candidate_id: item.probability for item in result.candidates},
                "reject_probability": result.reject_probability,
                "selected_capability": result.selected_candidate_id,
                "policy_result": policy.decision if policy else None,
                "action_id": outcome.get("action_id"),
                "approval_result": "PENDING" if outcome.get("action_id") else None,
                "execution_result": outcome["execution"],
            })
            outcomes.append(outcome)
        return outcomes

    def _execute(
        self, capability_id: str, snapshot: StateSnapshot, arguments: dict[str, Any],
        state_store: StateStore,
    ) -> Any:
        plugin = self.registry.implementation(capability_id)
        validate_arguments(plugin.descriptor.parameter_schema, arguments)
        if not state_store.is_current(snapshot.snapshot_id) or not plugin.available(snapshot.data):
            raise RuntimeError("EXECUTION_PRECONDITION_CHANGED")
        result = plugin.execute(snapshot.data, arguments)
        self.policy.record_execution(capability_id)
        return result

    def _prune_actions(self) -> None:
        """清理超时审批项及关联的临时状态引用。"""

        now = time.time()
        expired = [key for key, action in self.pending_actions.items()
                   if now - action.created_at > 3600]
        for key in expired:
            self.pending_actions.pop(key)
            self._action_states.pop(key, None)

    def approve(self, action_id: str, token: str | None = None) -> dict[str, Any]:
        """审批并执行仍满足前置条件的高风险动作。"""

        with self._lock:
            actor_id, permissions = self._require_approver(token)
            action = self.pending_actions[action_id]
            plugin = self.registry.implementation(action.capability_id)
            if plugin.descriptor.permission not in permissions:
                raise PermissionError("APPROVAL_PERMISSION_DENIED")
            if time.time() - action.created_at > 3600:
                self.pending_actions.pop(action_id)
                self._action_states.pop(action_id, None)
                raise ValueError("ACTION_EXPIRED")
            self.pending_actions.pop(action_id)
            state_store = self.sessions[action.session_id].state if action.session_id else self._action_states[action_id]
            snapshot = state_store.snapshot()
            if snapshot.snapshot_id != action.snapshot_id:
                self._action_states.pop(action_id, None)
                raise RuntimeError("STALE_ACTION")
            try:
                result = self._execute(action.capability_id, snapshot, action.arguments, state_store)
            finally:
                self._action_states.pop(action_id, None)
            self._record_approval(action_id, "APPROVED", result, actor_id)
            return {"action_id": action_id, "status": "EXECUTED", "result": result}

    def reject(self, action_id: str, token: str | None = None) -> dict[str, str]:
        """拒绝并移除待审批动作。"""

        with self._lock:
            actor_id, _ = self._require_approver(token)
            self.pending_actions.pop(action_id)
            self._action_states.pop(action_id, None)
            self._record_approval(action_id, "REJECTED", None, actor_id)
            return {"action_id": action_id, "status": "REJECTED"}

    def list_actions(self, token: str | None) -> list[dict[str, Any]]:
        """仅向审批者列出未过期的待审批动作。"""

        with self._lock:
            self._require_approver(token)
            self._prune_actions()
            return [
                {key: value for key, value in asdict(action).items() if key != "tenant_id"}
                for action in self.pending_actions.values()
            ]

    def list_audit(self, token: str | None, limit: int = 100) -> list[dict[str, Any]]:
        """仅向审批者返回最近审计记录。

        @param limit 返回条数，范围为 1 到 1000。
        """

        with self._lock:
            self._require_approver(token)
            if not 1 <= limit <= 1000:
                raise ValueError("limit 必须位于 [1,1000]")
            return [dict(event) for event in list(self.audit)[-limit:]]

    def _require_approver(self, token: str | None) -> tuple[str, frozenset[str]]:
        """验证服务端审批权限并返回身份、权限。

        @return 服务端令牌对应的 actor_id 与权限集合。
        """

        actor_id, permissions, _ = self.resolve_actor(token)
        if "approve" not in permissions:
            raise PermissionError("APPROVAL_PERMISSION_DENIED")
        return actor_id, permissions

    def _record_approval(
        self, action_id: str, result: str, execution: Any, actor_id: str
    ) -> None:
        """在原始决策审计记录上补齐人工审批结果。"""

        for event in reversed(self.audit):
            if event.get("action_id") == action_id:
                event["approval_result"] = result
                event["approval_actor_id"] = actor_id
                if execution is not None:
                    event["execution_result"] = execution
                return

    def list_plugins(self) -> dict[str, list[dict[str, object]]]:
        return self.registry.describe()

    def refresh_mcp(self) -> dict[str, list[str]]:
        """显式更新已配置 MCP Server 的 Tool Schema 缓存。"""

        with self._lock:
            return {client.server_id: client.refresh() for client in self._mcp_clients}

    def get_plugin(self, plugin_id: str) -> dict[str, object]:
        return self.registry.get(plugin_id)

    def get_status(self) -> dict[str, Any]:
        return {
            "engine": "online",
            "state_versions": {
                tenant: store.snapshot().version for tenant, store in self._state_stores.items()
            },
            "sessions": len(self.sessions),
            "pending_actions": len(self.pending_actions),
            "audit_events": len(self.audit),
            "metrics": self.engine.metrics(),
            "contract": self.contract.manifest(),
        }
