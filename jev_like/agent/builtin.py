from __future__ import annotations

from typing import Any

from .plugins import CapabilityDescriptor, CapabilityPlugin, DecisionDescriptor
from .registry import PluginRegistry


def register_builtin_plugins(registry: PluginRegistry) -> None:
    """注册最小 Runtime Recovery 示例插件，便于 API/UI 开箱验证。"""

    registry.register_decision(DecisionDescriptor(
        id="runtime.next_action",
        display_name="Runtime Next Action",
        description="根据运行指标和最近错误选择恢复动作。",
        trigger="TASK_FAILED",
        required_state=("runtime_metrics", "observations", "environment"),
        question="What should the runtime do next?",
        candidate_query="tag:recovery_action",
        priority=10,
    ))

    def declarative(action: str):
        return lambda state, arguments: {"action": action, "accepted": True, "arguments": arguments}

    capabilities = (
        CapabilityDescriptor(
            "runtime.reduce_batch", "Reduce Batch", "reduce the inference batch token budget",
            "降低 token budget 以缓解显存压力。", ("recovery_action", "gpu_memory"), "low",
        ),
        CapabilityDescriptor(
            "runtime.defer_work", "Defer Work", "defer non-urgent inference requests",
            "延后非紧急推理请求。", ("recovery_action", "gpu_memory"), "medium",
        ),
        CapabilityDescriptor(
            "runtime.restart_worker", "Restart Worker", "restart the unhealthy inference worker",
            "重启异常推理 Worker，需要人工审批。", ("recovery_action",), "high",
        ),
    )
    for descriptor in capabilities:
        registry.register_capability(CapabilityPlugin(
            descriptor,
            available=lambda state: True,
            execute=declarative(descriptor.id),
        ))
