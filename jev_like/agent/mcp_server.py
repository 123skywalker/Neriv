from __future__ import annotations

from typing import Any

from .service import AgentRequest, AgentService


def create_mcp_server(service: AgentService) -> Any:
    """把同一 AgentService 暴露为 MCP tools，不绕开策略与审批。"""

    try:
        from mcp.server import MCPServer
    except ImportError as error:
        raise RuntimeError("MCP Server 需要安装 mcp") from error

    server = MCPServer("neriv")

    @server.tool()
    def neriv_decide(session_id: str, decisions: list[dict[str, Any]],
                     execution_mode: str = "EVALUATE", timeout_ms: int = 30_000) -> dict[str, Any]:
        """根据显式会话执行一个或多个决策。"""

        return service.decide(AgentRequest.from_dict({
            "session_id": session_id, "decisions": decisions,
            "execution_mode": execution_mode, "timeout_ms": timeout_ms,
        }))

    @server.tool()
    def neriv_system_one(session_id: str, instructions: str, candidates: list[str],
                         decision_type: str = "CHOICE") -> dict[str, Any]:
        """以临时类型化问题调用 System-One 决策。"""

        return service.decide(AgentRequest.from_dict({
            "session_id": session_id, "decisions": [{"decision_id": "systemone", "ad_hoc": {
                "type": decision_type, "instructions": instructions, "candidates": candidates,
            }}],
        }))

    return server


def main() -> None:
    """从 checkpoint 启动独立 MCP Streamable HTTP 服务。"""

    import argparse
    from pathlib import Path

    from jev_like.server import build_service

    parser = argparse.ArgumentParser(description="启动 Neriv MCP Server")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    create_mcp_server(build_service(args.checkpoint, args.model, cpu=args.cpu)).run(
        transport="streamable-http"
    )


if __name__ == "__main__":
    main()
