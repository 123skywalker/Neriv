from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal


Risk = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class DecisionDescriptor:
    """Decision Plugin 的静态、可编译描述。"""

    id: str
    display_name: str
    description: str
    trigger: str
    required_state: tuple[str, ...]
    question: str
    candidate_query: str
    priority: int = 0
    version: str = "1"
    decision_type: str = "CHOICE"
    policy_ref: str = "runtime-recovery@v1"


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Capability Plugin 的模型描述与确定性安全元数据。"""

    id: str
    display_name: str
    model_text: str
    description: str
    tags: tuple[str, ...]
    risk: Risk = "low"
    permission: str = "execute"
    parameter_schema: dict[str, Any] = field(default_factory=dict)
    version: str = "1"
    source: Literal["LOCAL", "MCP"] = "LOCAL"
    source_revision: str = "local"


@dataclass(slots=True)
class CapabilityPlugin:
    """Capability 的本地运行实现；对象不会进入 Engine 热路径。"""

    descriptor: CapabilityDescriptor
    available: Callable[[dict[str, Any]], bool]
    execute: Callable[[dict[str, Any], dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class CompiledDecision:
    """启动阶段编译后的 Decision 表记录。"""

    numeric_id: int
    descriptor: DecisionDescriptor
    question_tokens: tuple[int, ...]
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class CompiledCapability:
    """启动阶段编译后的 Capability 表记录。"""

    numeric_id: int
    descriptor: CapabilityDescriptor
    candidate_tokens: tuple[int, ...]
    enabled: bool = True
