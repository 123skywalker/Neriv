from __future__ import annotations

from jev_like.model.contract import ModelContract

from .plugins import (
    CapabilityPlugin,
    CompiledCapability,
    CompiledDecision,
    DecisionDescriptor,
)
from .state import StateSnapshot


class PluginRegistry:
    """验证、编译并索引静态 Decision 与本地或 MCP Capability。"""

    def __init__(self, contract: ModelContract) -> None:
        self.contract = contract
        self._decisions: dict[str, CompiledDecision] = {}
        self._capabilities: dict[str, CompiledCapability] = {}
        self._implementations: dict[str, CapabilityPlugin] = {}
        self._tags: dict[str, set[str]] = {}

    def register_decision(self, descriptor: DecisionDescriptor) -> None:
        """校验并缓存静态 Question token。"""

        if descriptor.id in self._decisions or not descriptor.id.strip():
            raise ValueError(f"重复或空 Decision Plugin ID: {descriptor.id}")
        tokens = self.contract.question_tokens(descriptor.question)
        self._decisions[descriptor.id] = CompiledDecision(len(self._decisions), descriptor, tokens)

    def register_capability(self, plugin: CapabilityPlugin) -> None:
        """校验描述、缓存 Candidate token 并更新 TagIndex。"""

        descriptor = plugin.descriptor
        existing = self._capabilities.get(descriptor.id)
        if not descriptor.id.strip() or existing and (existing.descriptor.source != "MCP" or descriptor.source != "MCP"):
            raise ValueError(f"重复或空 Capability Plugin ID: {descriptor.id}")
        tokens = self.contract.candidate_tokens(descriptor.model_text)
        if existing:
            for ids in self._tags.values():
                ids.discard(descriptor.id)
        self._capabilities[descriptor.id] = CompiledCapability(
            existing.numeric_id if existing else len(self._capabilities), descriptor, tokens
        )
        self._implementations[descriptor.id] = plugin
        for tag in descriptor.tags:
            self._tags.setdefault(tag, set()).add(descriptor.id)

    def set_enabled(self, plugin_id: str, enabled: bool) -> None:
        """启用或禁用插件，保留其稳定 numeric_id。"""

        if plugin_id in self._decisions:
            record = self._decisions[plugin_id]
            self._decisions[plugin_id] = CompiledDecision(
                record.numeric_id, record.descriptor, record.question_tokens, enabled
            )
            return
        if plugin_id in self._capabilities:
            record = self._capabilities[plugin_id]
            self._capabilities[plugin_id] = CompiledCapability(
                record.numeric_id, record.descriptor, record.candidate_tokens, enabled
            )
            return
        raise KeyError(plugin_id)

    def decisions_for(self, trigger: str, selected: tuple[str, ...] = ()) -> list[CompiledDecision]:
        """按显式 ID 或事件 trigger 返回稳定顺序的启用 Decision。"""

        values = (
            [self._decisions[plugin_id] for plugin_id in dict.fromkeys(selected)]
            if selected
            else [record for record in self._decisions.values() if record.descriptor.trigger == trigger]
        )
        return sorted((record for record in values if record.enabled), key=lambda record: record.numeric_id)

    def resolve_candidates(
        self,
        query: str,
        snapshot: StateSnapshot,
        permissions: set[str] | None,
    ) -> list[CompiledCapability]:
        """按固定流水线解析、过滤、排序，超出模型上限时显式拒绝。"""

        if not query.startswith("tag:"):
            raise ValueError(f"不支持的 candidate_query: {query}")
        ids = self._tags.get(query.removeprefix("tag:"), set())
        resolved = []
        for plugin_id in sorted(ids):
            record = self._capabilities[plugin_id]
            plugin = self._implementations[plugin_id]
            if (record.enabled and (permissions is None or record.descriptor.permission in permissions)
                    and plugin.available(snapshot.data)):
                resolved.append(record)
        if len(resolved) > 16:
            raise ValueError("TOO_MANY_CANDIDATES")
        return resolved

    def implementation(self, capability_id: str) -> CapabilityPlugin:
        """返回 Action Executor 使用的本地实现。"""

        return self._implementations[capability_id]

    def describe(self) -> dict[str, list[dict[str, object]]]:
        """生成 REST 与插件驱动 UI 共用的 Schema。"""

        decisions = [
            {
                "id": record.descriptor.id,
                "display_name": record.descriptor.display_name,
                "description": record.descriptor.description,
                "question": record.descriptor.question,
                "trigger": record.descriptor.trigger,
                "type": record.descriptor.decision_type,
                "version": record.descriptor.version,
                "required_state": record.descriptor.required_state,
                "enabled": record.enabled,
            }
            for record in self._decisions.values()
        ]
        capabilities = [
            {
                "id": record.descriptor.id,
                "display_name": record.descriptor.display_name,
                "description": record.descriptor.description,
                "tags": record.descriptor.tags,
                "risk": record.descriptor.risk,
                "permission": record.descriptor.permission,
                "source": record.descriptor.source,
                "version": record.descriptor.version,
                "source_revision": record.descriptor.source_revision,
                "enabled": record.enabled,
            }
            for record in self._capabilities.values()
        ]
        return {"decisions": decisions, "capabilities": capabilities}

    def get(self, plugin_id: str) -> dict[str, object]:
        """读取单个插件的 UI/API 描述。"""

        schema = self.describe()
        for record in (*schema["decisions"], *schema["capabilities"]):
            if record["id"] == plugin_id:
                return record
        raise KeyError(plugin_id)
