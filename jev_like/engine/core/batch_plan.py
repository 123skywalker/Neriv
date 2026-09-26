from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class Stage(IntEnum):
    """GPU 执行阶段，数值同时表示基础调度优先级。"""

    STATE = 1
    QUESTION = 2
    CANDIDATE = 3


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Scheduler 与 ModelRunner 之间唯一的批执行接口。"""

    stage: Stage
    node_ids: tuple[int, ...]
    token_data: torch.Tensor
    token_indptr: torch.Tensor
    prefix_ids: tuple[int, ...]

    @property
    def sequence_count(self) -> int:
        return len(self.node_ids)

    @property
    def token_count(self) -> int:
        return int(self.token_data.numel())

