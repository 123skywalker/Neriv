from types import SimpleNamespace

from jev_like.agent.mcp_client import McpCapabilityClient
from jev_like.agent.registry import PluginRegistry
from jev_like.model.contract import ModelContract


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) % 53 + 1 for char in text]


def test_mcp_discovery_is_cached_until_explicit_refresh() -> None:
    registry = PluginRegistry(ModelContract(_Tokenizer(), "qwen", "v1", "v1"))
    client = McpCapabilityClient(registry, "tools", "http://localhost:8001/mcp")
    calls = []

    async def session(operation, tool="", arguments=None):
        calls.append(operation)
        return SimpleNamespace(tools=[SimpleNamespace(name="search", description="Search documents",
                                                     input_schema={"type": "object", "properties": {}})])

    client._session = session
    assert client.refresh() == ["mcp:tools:search"]
    assert registry.get("mcp:tools:search")["source"] == "MCP"
    assert calls == ["list"]
