"""Neriv 插件化 Agent Runtime。"""

from .api import create_app
from .plugins import CapabilityDescriptor, CapabilityPlugin, DecisionDescriptor
from .registry import PluginRegistry
from .service import AgentRequest, AgentService, DecisionCall
from .sdk import NerivClient
from .mcp_client import McpCapabilityClient
from .mcp_server import create_mcp_server
from .state import StateSnapshot, StateStore

__all__ = [
    "AgentRequest",
    "AgentService",
    "DecisionCall",
    "NerivClient",
    "McpCapabilityClient",
    "create_mcp_server",
    "CapabilityDescriptor",
    "CapabilityPlugin",
    "DecisionDescriptor",
    "PluginRegistry",
    "StateSnapshot",
    "StateStore",
    "create_app",
]
