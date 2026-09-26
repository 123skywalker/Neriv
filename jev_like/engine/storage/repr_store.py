from __future__ import annotations

import heapq

import torch


class _SlotBuffer:
    def __init__(self, capacity: int, width: int, device: torch.device, dtype: torch.dtype) -> None:
        self.tensor = torch.empty((capacity, width), device=device, dtype=dtype)
        self.free = list(range(capacity))
        heapq.heapify(self.free)
        self.used: set[int] = set()

    def allocate(self, count: int) -> list[int]:
        if count > len(self.free):
            raise MemoryError(f"表示槽位不足: 请求 {count}，空闲 {len(self.free)}")
        slots = [heapq.heappop(self.free) for _ in range(count)]
        self.used.update(slots)
        return slots

    def release(self, slots: list[int]) -> None:
        for slot in slots:
            if slot not in self.used:
                raise ValueError(f"表示槽位 {slot} 未分配或已释放")
            self.used.remove(slot)
            heapq.heappush(self.free, slot)


class ReprStore:
    """预分配 Query/Candidate 表示缓冲区，运行时只传递 slot ID。"""

    def __init__(self, query_capacity: int, candidate_capacity: int, width: int,
                 device: torch.device | str, dtype: torch.dtype = torch.float32) -> None:
        target = torch.device(device)
        self._queries = _SlotBuffer(query_capacity, width, target, dtype)
        self._candidates = _SlotBuffer(candidate_capacity, width, target, dtype)

    def allocate_query_slots(self, count: int) -> list[int]:
        return self._queries.allocate(count)

    def allocate_candidate_slots(self, count: int) -> list[int]:
        return self._candidates.allocate(count)

    def write_query(self, slot: int, value: torch.Tensor) -> None:
        self._queries.tensor[slot].copy_(value)

    def write_candidate(self, slot: int, value: torch.Tensor) -> None:
        self._candidates.tensor[slot].copy_(value)

    def read_queries(self, slots: list[int]) -> torch.Tensor:
        return self._queries.tensor[slots]

    def read_candidates(self, slots: list[int]) -> torch.Tensor:
        return self._candidates.tensor[slots]

    def free_query_slots(self, slots: list[int]) -> None:
        self._queries.release(slots)

    def free_candidate_slots(self, slots: list[int]) -> None:
        self._candidates.release(slots)

    @property
    def free_query_count(self) -> int:
        return len(self._queries.free)

    @property
    def free_candidate_count(self) -> int:
        return len(self._candidates.free)

