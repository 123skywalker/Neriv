import torch

from jev_like.engine.kv import KVCacheManager
from jev_like.engine.storage import PagedKVStore


def _store(pages: int) -> PagedKVStore:
    return PagedKVStore(pages, 4, 1, 1, 2, "cpu", torch.float32)


def test_fork_copies_partial_tail_page() -> None:
    store = _store(4)
    page = store.allocate_pages(1)[0]
    manager = KVCacheManager(store)
    state = manager.create_prefix([page], 3)
    question = manager.fork_prefix(state)
    assert manager.page_ids(question) != manager.page_ids(state)
    assert store.free_page_count == 2
    manager.release(state)
    manager.release(question)
    assert store.free_page_count == 4


def test_aligned_prefix_shares_page_until_all_branches_release() -> None:
    store = _store(2)
    page = store.allocate_pages(1)[0]
    manager = KVCacheManager(store)
    state = manager.create_prefix([page], 4)
    question = manager.fork_prefix(state)
    assert manager.page_ids(question) == manager.page_ids(state)
    manager.release(state)
    assert store.free_page_count == 1
    manager.release(question)
    assert store.free_page_count == 2


def test_state_cache_reuses_only_full_pages_with_scope_isolation() -> None:
    store = _store(5)
    manager = KVCacheManager(store)
    source = manager.create_prefix([], 0)
    manager.reserve_append(source, 6)
    tokens = (1, 2, 3, 4, 5, 6)
    manager.publish_state_full_blocks(source, tokens, "tenant-a", "epoch")
    source_page = manager.page_ids(source)[0]
    store.tensor[:, source_page].fill_(3)
    manager.release(source)
    hit, count = manager.lookup_state_prefix(tokens + (7,), "tenant-a", "epoch")
    assert count == 4
    assert manager.seq_len(hit) == 4
    assert torch.all(store.tensor[:, manager.page_ids(hit)[0]] == 3)
    miss, count = manager.lookup_state_prefix(tokens, "tenant-b", "epoch")
    assert count == 0
    other_epoch, count = manager.lookup_state_prefix(tokens, "tenant-a", "new-epoch")
    assert count == 0
    manager.release(hit)
    manager.release(miss)
    manager.release(other_epoch)
    assert manager.evict_cached_until_free(5) == 1
    assert store.free_page_count == 5


def test_state_cache_never_evicts_active_pages() -> None:
    store = _store(2)
    manager = KVCacheManager(store)
    source = manager.create_prefix([], 0)
    manager.reserve_append(source, 4)
    manager.publish_state_full_blocks(source, (1, 2, 3, 4), "scope", "epoch")
    manager.release(source)
    active, count = manager.lookup_state_prefix((1, 2, 3, 4), "scope", "epoch")
    assert count == 4
    assert manager.evict_cached_until_free(2) == 0
    manager.release(active)
    assert manager.evict_cached_until_free(2) == 1


def test_active_cached_pages_are_counted_once_across_prefixes() -> None:
    store = _store(2)
    manager = KVCacheManager(store)
    source = manager.create_prefix([], 0)
    manager.reserve_append(source, 4)
    manager.publish_state_full_blocks(source, (1, 2, 3, 4), "scope", "epoch")
    manager.release(source)
    assert manager.active_cached_page_count == 0
    first, _ = manager.lookup_state_prefix((1, 2, 3, 4), "scope", "epoch")
    second, _ = manager.lookup_state_prefix((1, 2, 3, 4), "scope", "epoch")
    assert manager.active_cached_page_count == 1
    manager.release(first)
    assert manager.active_cached_page_count == 1
    manager.release(second)
    assert manager.active_cached_page_count == 0


def test_duplicate_miss_publish_does_not_transfer_uncached_page() -> None:
    store = _store(3)
    manager = KVCacheManager(store)
    first = manager.create_prefix([], 0)
    second = manager.create_prefix([], 0)
    manager.reserve_append(first, 4)
    manager.reserve_append(second, 4)
    tokens = (1, 2, 3, 4)
    assert manager.publish_state_full_blocks(first, tokens, "scope", "epoch") == 1
    assert manager.publish_state_full_blocks(second, tokens, "scope", "epoch") == 0
    assert manager.active_cached_page_count == 1
    manager.release(first)
    assert manager.active_cached_page_count == 0
    assert store.free_page_count == 1
    manager.release(second)
    assert store.free_page_count == 2
