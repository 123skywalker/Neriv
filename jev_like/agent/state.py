from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """一次批量决策使用的不可变状态副本。"""

    snapshot_id: str
    version: int
    data: dict[str, Any]


class StateStore:
    """单写结构化状态存储，并负责批量字段投影。"""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._version = 0
        self._lock = RLock()

    def update(self, values: dict[str, Any]) -> StateSnapshot:
        """原子合并顶层状态并生成新版本快照。"""

        with self._lock:
            self._data.update(copy.deepcopy(values))
            self._version += 1
            return self.snapshot()

    def snapshot(self) -> StateSnapshot:
        """返回与后续状态更新隔离的深拷贝。"""

        with self._lock:
            return StateSnapshot(str(self._version), self._version, copy.deepcopy(self._data))

    @staticmethod
    def project(snapshot: StateSnapshot, fields: set[str]) -> str:
        """按本次请求选定的字段集生成 Canonical State。"""

        projected = {field: snapshot.data.get(field) for field in sorted(fields)}
        return json.dumps(projected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def is_current(self, snapshot_id: str) -> bool:
        """检查 Engine 结果是否仍属于当前状态版本。"""

        with self._lock:
            return snapshot_id == str(self._version)
