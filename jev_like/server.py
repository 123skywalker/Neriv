from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from jev_like.agent import AgentService, create_app
from jev_like.engine import EngineConfig, NerivEngine
from jev_like.model.decision_model import load_training_checkpoint


def build_service(
    checkpoint: Path,
    model_path: Path | None = None,
    transformer_backend: str = "auto",
    engine_backend: str = "auto",
    cpu: bool = False,
    mcp_servers: dict[str, str] | None = None,
    auth_tokens: dict[str, dict] | None = None,
    prefix_cache: bool = True,
):
    """从 checkpoint 构造 REST 与 MCP 共用的 AgentService。"""

    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model, contract = load_training_checkpoint(
        checkpoint, model_path, attention_backend=transformer_backend, dtype=dtype
    )
    fingerprint = hashlib.sha256()
    with (checkpoint / "model.pt").open("rb") as weights:
        for chunk in iter(lambda: weights.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    engine = NerivEngine(model, EngineConfig(
        attention_backend=engine_backend, prefix_cache=prefix_cache,
        cache_epoch=fingerprint.hexdigest(),
    ), device=device)
    return AgentService(engine, contract, mcp_servers=mcp_servers, auth_tokens=auth_tokens)


def build_app(
    checkpoint: Path,
    model_path: Path | None = None,
    transformer_backend: str = "auto",
    engine_backend: str = "auto",
    cpu: bool = False,
    mcp_servers: dict[str, str] | None = None,
    auth_tokens: dict[str, dict] | None = None,
    prefix_cache: bool = True,
):
    """创建共享 AgentService 的 REST API 与 Web Console。"""

    return create_app(build_service(
        checkpoint, model_path, transformer_backend, engine_backend, cpu, mcp_servers, auth_tokens,
        prefix_cache,
    ))


def main() -> None:
    """启动 Neriv Agent API 与 `/ui` 控制台。"""

    parser = argparse.ArgumentParser(description="启动 Neriv Agent Service")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--attention-backend", default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--engine-backend", default="auto", choices=["auto", "flashinfer", "reference"]
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-prefix-cache", action="store_true", help="关闭跨请求 State 整页缓存")
    parser.add_argument("--auth-file", type=Path, help="服务端令牌权限表 JSON 文件")
    parser.add_argument("--mcp-server", action="append", default=[], metavar="ID=URL",
                        help="启动时导入外部 MCP Tool，可重复指定")
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("服务端需要安装 `pip install -e .[server]`") from error
    mcp_servers = dict(value.split("=", 1) for value in args.mcp_server)
    auth_tokens = json.loads(args.auth_file.read_text(encoding="utf-8")) if args.auth_file else None
    app = build_app(args.checkpoint, args.model, args.attention_backend, args.engine_backend,
                    args.cpu, mcp_servers, auth_tokens, not args.no_prefix_cache)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
