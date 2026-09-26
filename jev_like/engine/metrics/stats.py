from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass(slots=True)
class EngineStats:
    """无外部依赖的计数与延迟聚合器。"""

    counters: Counter[str] = field(default_factory=Counter)
    latency_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    latency_count: Counter[str] = field(default_factory=Counter)

    def observe(self, name: str, seconds: float) -> None:
        self.latency_sum[name] += seconds
        self.latency_count[name] += 1

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def snapshot(self) -> dict[str, float | int]:
        result: dict[str, float | int] = dict(self.counters)
        for name, total in self.latency_sum.items():
            result[f"{name}_mean_seconds"] = total / self.latency_count[name]
        return result
