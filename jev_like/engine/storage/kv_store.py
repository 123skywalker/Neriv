from __future__ import annotations

import heapq
from typing import Protocol

import torch


class PhysicalKVStore(Protocol):
    """物理 Paged KV Store 的最小协议。"""

    page_size: int

    @property
    def free_page_count(self) -> int: ...
    def allocate_pages(self, count: int) -> list[int]: ...
    def free_pages(self, page_ids: list[int]) -> None: ...
    def copy_page(self, source: int, target: int) -> None: ...
    def layer(self, layer_index: int) -> torch.Tensor: ...
    def page_table(self, page_ids: list[int]) -> torch.Tensor: ...


class PagedKVStore:
    """启动时一次性分配、运行中只复用页 ID 的 GPU KV Tensor。"""

    def __init__(
        self,
        page_count: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if page_count <= 0 or page_size <= 0:
            raise ValueError("page_count 和 page_size 必须大于 0")
        self.page_size = page_size
        self.tensor = torch.empty(
            (num_layers, page_count, 2, page_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._allocated = [False] * page_count
        self._free = list(range(page_count))
        heapq.heapify(self._free)

    @property
    def free_page_count(self) -> int:
        return len(self._free)

    def allocate_pages(self, count: int) -> list[int]:
        if count < 0 or count > len(self._free):
            raise MemoryError(f"KV 页不足: 请求 {count}，空闲 {len(self._free)}")
        pages = [heapq.heappop(self._free) for _ in range(count)]
        for page_id in pages:
            self._allocated[page_id] = True
        return pages

    def free_pages(self, page_ids: list[int]) -> None:
        for page_id in page_ids:
            if not 0 <= page_id < len(self._allocated) or not self._allocated[page_id]:
                raise ValueError(f"KV 页 {page_id} 未分配或已释放")
            self._allocated[page_id] = False
            heapq.heappush(self._free, page_id)

    def copy_page(self, source: int, target: int) -> None:
        """复制一个跨全部 Transformer Layer 的物理页。"""

        if not self._allocated[source] or not self._allocated[target]:
            raise ValueError("复制页必须已分配")
        self.tensor[:, target].copy_(self.tensor[:, source])

    def layer(self, layer_index: int) -> torch.Tensor:
        """返回 FlashInfer NHD 布局的单层 Paged KV。"""

        return self.tensor[layer_index]

    def page_table(self, page_ids: list[int]) -> torch.Tensor:
        """把逻辑页 ID 转换为 GPU int32 Page Table。"""

        return torch.tensor(page_ids, device=self.tensor.device, dtype=torch.int32)
