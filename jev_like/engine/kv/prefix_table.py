from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PrefixRecord:
    page_ids: tuple[int, ...]
    seq_len: int
    active: bool = True


class PrefixTable:
    """稳定整数 ID 索引的 Prefix 元数据表。"""

    def __init__(self) -> None:
        self.records: list[PrefixRecord] = []
        self._free: list[int] = []

    def create(self, page_ids: list[int], seq_len: int) -> int:
        if self._free:
            prefix_id = self._free.pop()
            self.records[prefix_id] = PrefixRecord(tuple(page_ids), seq_len)
            return prefix_id
        prefix_id = len(self.records)
        self.records.append(PrefixRecord(tuple(page_ids), seq_len))
        return prefix_id

    def get(self, prefix_id: int) -> PrefixRecord:
        try:
            record = self.records[prefix_id]
        except IndexError as error:
            raise ValueError(f"未知 prefix_id: {prefix_id}") from error
        if not record.active:
            raise ValueError(f"prefix_id 已释放: {prefix_id}")
        return record

    def deactivate(self, prefix_id: int) -> None:
        """释放记录槽，后续 create 可复用同一整数 ID。"""

        record = self.get(prefix_id)
        record.active = False
        self._free.append(prefix_id)
