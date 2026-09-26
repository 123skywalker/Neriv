from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from .plugins import CapabilityDescriptor, CapabilityPlugin
from .registry import PluginRegistry


class McpCapabilityClient:
    """启动或刷新时发现远程工具，并把调用接入同一 Capability Guard。"""

    def __init__(self, registry: PluginRegistry, server_id: str, url: str) -> None:
        self.registry = registry
        self.server_id = server_id
        self.url = url
        self._imported: set[str] = set()

    async def _session(self, operation: str, tool: str = "", arguments: dict[str, Any] | None = None) -> Any:
        try:
            from mcp import Client
        except ImportError as error:
            raise RuntimeError("MCP Client 需要安装 mcp") from error
        async with Client(self.url) as client:
            return await (client.list_tools() if operation == "list" else client.call_tool(tool, arguments or {}))

    def refresh(self) -> list[str]:
        """仅在显式刷新时执行 tools/list，并以 schema 哈希标记来源版本。"""

        listed = asyncio.run(self._session("list"))
        imported = []
        for tool in listed.tools:
            schema = tool.input_schema or {}
            revision = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12]
            capability_id = f"mcp:{self.server_id}:{tool.name}"
            descriptor = CapabilityDescriptor(
                id=capability_id, display_name=tool.name,
                model_text=tool.description or tool.name, description=tool.description or "",
                tags=("mcp", self.server_id), risk="high", permission="execute",
                parameter_schema=schema, version="1", source="MCP", source_revision=revision,
            )
            self.registry.register_capability(CapabilityPlugin(
                descriptor, available=lambda state: True,
                execute=lambda state, args, name=tool.name: self.call(name, args),
            ))
            imported.append(capability_id)
        for stale_id in self._imported - set(imported):
            self.registry.set_enabled(stale_id, False)
        self._imported = set(imported)
        return imported

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """调用已发现的远程工具；执行入口仍由 Agent Policy/Guard 控制。"""

        result = asyncio.run(self._session("call", tool, arguments))
        if result.is_error:
            raise RuntimeError(f"MCP_TOOL_ERROR:{tool}")
        return result.structured_content or [item.text for item in result.content if hasattr(item, "text")]
