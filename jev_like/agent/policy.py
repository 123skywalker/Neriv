from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from jev_like.engine.request import DecisionResult

from .plugins import CapabilityDescriptor


class PolicyDecision(str, Enum):
    """Policy Gate 的有限结果集合。"""

    ALLOW = "ALLOW"
    REJECT = "REJECT"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """确定性 Gate 结果与原因。"""

    decision: PolicyDecision
    reason: str


@dataclass(frozen=True, slots=True)
class PolicyDescriptor:
    """以 Registry ID 引用的确定性决策阈值与执行模式。"""

    id: str
    version: str = "v1"
    min_confidence: float = 0.45
    max_reject_probability: float = 0.5
    allowed_modes: tuple[str, ...] = ("EVALUATE", "PROPOSE", "EXECUTE")
    max_retries: int = 1
    retry_backoff_ms: int = 50


class PolicyGate:
    """按固定顺序检查置信度、权限、风险与冷却。"""

    def __init__(self, cooldown_seconds: float = 1.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_execution: dict[str, float] = {}
        default = PolicyDescriptor("runtime-recovery@v1")
        self._policies = {default.id: default}

    def register(self, descriptor: PolicyDescriptor) -> None:
        """注册一个可由 Decision 引用的策略版本。"""

        if descriptor.id in self._policies:
            raise ValueError(f"重复策略 ID: {descriptor.id}")
        self._policies[descriptor.id] = descriptor

    def get(self, policy_ref: str) -> PolicyDescriptor:
        """解析策略引用；未知策略应 fail closed。"""

        return self._policies[policy_ref]

    def check(
        self,
        result: DecisionResult,
        policy_ref: str,
        capability: CapabilityDescriptor | None,
        snapshot_current: bool,
        permissions: set[str],
        mode: str,
    ) -> PolicyResult:
        """执行无副作用的 Policy Gate。

        @return ALLOW、REJECT、REQUIRE_APPROVAL 或 FALLBACK。
        """

        policy = self.get(policy_ref)
        if mode not in policy.allowed_modes:
            return PolicyResult(PolicyDecision.REJECT, "MODE_DENIED")
        if not snapshot_current:
            return PolicyResult(PolicyDecision.REJECT, "STALE_SNAPSHOT")
        if capability is None or result.selected_candidate_id is None:
            return PolicyResult(PolicyDecision.FALLBACK, "NO_CANDIDATE")
        if result.reject_probability > policy.max_reject_probability:
            return PolicyResult(PolicyDecision.FALLBACK, "REJECT_PROBABILITY")
        if result.max_probability < policy.min_confidence:
            return PolicyResult(PolicyDecision.FALLBACK, "LOW_CONFIDENCE")
        if capability.permission not in permissions:
            return PolicyResult(PolicyDecision.REJECT, "PERMISSION_DENIED")
        if capability.risk == "high":
            return PolicyResult(PolicyDecision.REQUIRE_APPROVAL, "HIGH_RISK")
        last = self._last_execution.get(capability.id, 0.0)
        if time.monotonic() - last < self.cooldown_seconds:
            return PolicyResult(PolicyDecision.REJECT, "COOLDOWN")
        return PolicyResult(PolicyDecision.ALLOW, "POLICY_OK")

    def record_execution(self, capability_id: str) -> None:
        """在成功执行后启动 Capability 冷却计时。"""

        self._last_execution[capability_id] = time.monotonic()


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    """校验 Capability 的必填字段、基本 JSON 类型和枚举。"""

    required = schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"缺少 Capability 参数: {missing}")
    if schema.get("additionalProperties") is False:
        allowed = set(schema.get("properties", {}))
        extra = sorted(set(arguments) - allowed)
        if extra:
            raise ValueError(f"未知 Capability 参数: {extra}")
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
             "object": dict, "array": list}
    for name, value in arguments.items():
        spec = schema.get("properties", {}).get(name, {})
        expected = types.get(spec.get("type"))
        if expected and (not isinstance(value, expected) or isinstance(value, bool) and spec.get("type") in {"integer", "number"}):
            raise ValueError(f"Capability 参数类型错误: {name}")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"Capability 参数不在允许枚举中: {name}")
