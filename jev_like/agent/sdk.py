from __future__ import annotations

from typing import Any

class NerivClient:
    """仅封装 Agent REST 边界的同步 Python SDK。"""

    def __init__(self, base_url: str, timeout: float = 30.0, access_token: str | None = None) -> None:
        """创建客户端；权限由服务端令牌决定，请求体声明无效。"""

        try:
            import httpx
        except ImportError as error:
            raise RuntimeError("Python SDK 需要安装 httpx") from error
        headers = {"Authorization": f"Bearer {access_token}"} if access_token else None
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers)

    def __enter__(self) -> "NerivClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """关闭 HTTP 连接池。"""

        self._client.close()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._client.request(method, path, json=payload)
        response.raise_for_status()
        return response.json()

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        """提交类型化决策请求。"""

        return self._request("POST", "/api/v1/decide", payload)

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        """创建会话。"""

        return self._request("POST", "/api/v1/sessions", payload)

    def update_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """更新会话并获得新状态版本。"""

        return self._request("PATCH", f"/api/v1/sessions/{session_id}", payload)

    def plugins(self) -> dict[str, Any]:
        """读取可用插件描述。"""

        return self._request("GET", "/api/v1/plugins")

    def system_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        """调用只返回类型化概率的 System-One 接口。"""

        return self._request("POST", "/v1/systemone", payload)

    def status(self) -> dict[str, Any]:
        """读取运行状态与 Engine 指标。"""

        return self._request("GET", "/api/v1/status")
