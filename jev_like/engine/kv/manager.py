from __future__ import annotations

import hashlib
import json
from threading import RLock
from typing import Iterator

from jev_like.engine.storage.kv_store import PhysicalKVStore

from .prefix_table import PrefixTable


class KVCacheManager:
    """只管理 Prefix/Page 逻辑所有权；不接触 K/V Tensor 内容。"""

    def __init__(self, store: PhysicalKVStore) -> None:
        self.store = store
        self.prefixes = PrefixTable()
        self._page_refs: dict[int, int] = {}
        self._cached: dict[str, tuple[int, int]] = {}
        self._access = 0
        self._lock = RLock()

    def _block_keys(self, tokens: tuple[int, ...], scope_id: str, epoch: str) -> Iterator[str]:
        """按 scope 和完整页 token 生成稳定的链式 SHA-256 键。"""

        previous = ""
        for offset in range(0, len(tokens) - len(tokens) % self.store.page_size, self.store.page_size):
            block = tokens[offset:offset + self.store.page_size]
            payload = json.dumps((epoch, scope_id, previous, block), separators=(",", ":"))
            previous = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            yield previous

    def lookup_state_prefix(self, tokens: tuple[int, ...], scope_id: str, epoch: str) -> tuple[int, int]:
        """保留同一 scope 下最长的连续完整页前缀。

        @return 新 Prefix ID 与命中的 token 数。
        """

        with self._lock:
            pages = []
            for key in self._block_keys(tokens, scope_id, epoch):
                cached = self._cached.get(key)
                if cached is None:
                    break
                self._access += 1
                self._cached[key] = (cached[0], self._access)
                pages.append(cached[0])
            length = len(pages) * self.store.page_size
            return self.create_prefix(pages, length), length

    def publish_state_full_blocks(self, prefix_id: int, tokens: tuple[int, ...],
                                  scope_id: str, epoch: str) -> int:
        """只固定已完成 State 的整页；尾部不足一页不缓存。

        @return 本次进入活跃缓存计数的物理页数。
        """

        with self._lock:
            record = self.prefixes.get(prefix_id)
            cached_pages = {page_id for page_id, _ in self._cached.values()}
            published = 0
            for page_id, key in zip(record.page_ids, self._block_keys(tokens[:record.seq_len], scope_id, epoch)):
                if key not in self._cached:
                    self._access += 1
                    self._cached[key] = (page_id, self._access)
                    self._page_refs[page_id] += 1
                    if page_id not in cached_pages:
                        cached_pages.add(page_id)
                        published += 1
            return published

    def evict_cached_until_free(self, free_pages: int) -> int:
        """按 LRU 回收无活跃引用的缓存页。"""

        with self._lock:
            removed = 0
            while self.store.free_page_count < free_pages:
                available = ((key, page_id, access) for key, (page_id, access) in self._cached.items()
                             if self._page_refs[page_id] == 1)
                oldest = min(available, key=lambda item: item[2], default=None)
                if oldest is None:
                    break
                key, page_id, _ = oldest
                del self._cached[key]
                self._release_pages((page_id,))
                removed += 1
            return removed

    @property
    def cached_page_count(self) -> int:
        with self._lock:
            return len(self._cached)

    @property
    def active_cached_page_count(self) -> int:
        """统计被请求引用的缓存物理页，同一页仅计一次。

        @return 当前不可驱逐的缓存物理页数。
        """

        with self._lock:
            return len({page_id for page_id, _ in self._cached.values() if self._page_refs[page_id] > 1})

    def create_prefix(self, page_ids: list[int], seq_len: int) -> int:
        with self._lock:
            for page_id in page_ids:
                self._page_refs[page_id] = self._page_refs.get(page_id, 0) + 1
            return self.prefixes.create(page_ids, seq_len)

    def fork_prefix(self, parent_prefix_id: int) -> int:
        with self._lock:
            parent = self.prefixes.get(parent_prefix_id)
            pages = list(parent.page_ids)
            if pages and parent.seq_len % self.store.page_size:
                self.evict_cached_until_free(1)
                private_page = self.store.allocate_pages(1)[0]
                self.store.copy_page(pages[-1], private_page)
                pages[-1] = private_page
            return self.create_prefix(pages, parent.seq_len)

    def reserve_append(self, prefix_id: int, token_count: int) -> None:
        """为 Prefix Append 分配足够物理页并推进逻辑长度。"""

        with self._lock:
            record = self.prefixes.get(prefix_id)
            total = record.seq_len + token_count
            needed = (total + self.store.page_size - 1) // self.store.page_size
            extra = needed - len(record.page_ids)
            if extra > 0:
                self.evict_cached_until_free(extra)
                pages = self.store.allocate_pages(extra)
                for page_id in pages:
                    self._page_refs[page_id] = 1
                record.page_ids = (*record.page_ids, *pages)
            record.seq_len = total

    def release(self, prefix_id: int) -> None:
        with self._lock:
            record = self.prefixes.get(prefix_id)
            self._release_pages(record.page_ids)
            self.prefixes.deactivate(prefix_id)

    def _release_pages(self, page_ids: tuple[int, ...]) -> None:
        free: list[int] = []
        for page_id in page_ids:
            self._page_refs[page_id] -= 1
            if self._page_refs[page_id] == 0:
                del self._page_refs[page_id]
                free.append(page_id)
        if free:
            self.store.free_pages(free)

    def page_ids(self, prefix_id: int) -> list[int]:
        with self._lock:
            return list(self.prefixes.get(prefix_id).page_ids)

    def seq_len(self, prefix_id: int) -> int:
        with self._lock:
            return self.prefixes.get(prefix_id).seq_len
