from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jev_like.engine.backend import FlashInferPagedBackend, PagedBatch, ReferencePagedBackend
from jev_like.engine.kv import KVCacheManager
from jev_like.engine.storage import PagedKVStore
from jev_like.model.decision_model import NerivDecisionModel


def _append(backend, store, manager, prefix_id: int, tokens: tuple[int, ...]) -> torch.Tensor:
    """向单个 Prefix 追加 token，并返回最后一个 token 表示。"""

    old_length = manager.seq_len(prefix_id)
    manager.reserve_append(prefix_id, len(tokens))
    pages = manager.page_ids(prefix_id)
    device = backend.device
    batch = PagedBatch(
        torch.tensor(tokens, dtype=torch.long, device=device),
        torch.tensor([0, len(tokens)], dtype=torch.int32, device=device),
        torch.tensor([0, len(pages)], dtype=torch.int32, device=device),
        torch.tensor(pages, dtype=torch.int32, device=device),
        torch.tensor([manager.seq_len(prefix_id) % store.page_size or store.page_size], dtype=torch.int32, device=device),
        torch.tensor([old_length], dtype=torch.int32, device=device),
        torch.zeros(len(tokens), dtype=torch.int32, device=device),
        torch.arange(old_length, old_length + len(tokens), dtype=torch.int32, device=device),
    )
    return backend.run(batch, store)[0]


def _paged_path(model, backend_type, segments: tuple[tuple[int, ...], ...]) -> list[torch.Tensor]:
    """以 State→Question→Candidate 的 Prefix Fork 路径执行 Paged Backend。"""

    config = model.backbone.config
    dtype = next(model.backbone.parameters()).dtype
    store = PagedKVStore(128, 16, config.num_hidden_layers, config.num_key_value_heads,
                         config.head_dim, "cuda", dtype)
    manager = KVCacheManager(store)
    backend = backend_type(model.backbone, "cuda")
    prefixes = [manager.create_prefix([], 0)]
    outputs: list[torch.Tensor] = []
    for index, tokens in enumerate(segments):
        if index:
            prefixes.append(manager.fork_prefix(prefixes[-1]))
        outputs.append(_append(backend, store, manager, prefixes[-1], tokens))
    for prefix_id in reversed(prefixes):
        manager.release(prefix_id)
    if store.free_page_count != 128:
        raise AssertionError("Paged KV 页生命周期泄漏")
    return outputs


@torch.inference_mode()
def verify(model_path: Path, include_flashinfer: bool) -> dict[str, float | bool]:
    """对拍 Full Forward、Reference Paged 与 FlashInfer Paged。"""

    model, contract = NerivDecisionModel.from_pretrained(
        model_path, attention_backend="sdpa", dtype=torch.bfloat16, max_sequence_tokens=256
    )
    model.cuda().eval()
    segments = (
        contract.state_tokens("GPU memory is high."),
        contract.question_tokens("What should the runtime do next?"),
        contract.candidate_tokens("reduce the batch token budget"),
    )
    reference = _paged_path(model, ReferencePagedBackend, segments)
    full_diffs = []
    full_relative = []
    full_cosine = []
    combined: tuple[int, ...] = ()
    for tokens, representation in zip(segments, reference):
        combined += tokens
        ids = torch.tensor([combined], dtype=torch.long, device="cuda")
        expected = model.backbone(input_ids=ids, use_cache=False, return_dict=False)[0][0, -1]
        full_diffs.append(float((representation - expected).abs().max()))
        full_relative.append(float((representation - expected).float().norm() / expected.float().norm()))
        full_cosine.append(float(torch.nn.functional.cosine_similarity(
            representation.float(), expected.float(), dim=0
        )))
    report: dict[str, float | bool] = {
        "reference_full_max_abs": max(full_diffs),
        "reference_full_max_relative_l2": max(full_relative),
        "reference_full_min_cosine": min(full_cosine),
        "page_lifetime": True,
    }
    if report["reference_full_max_relative_l2"] > 0.05 or report["reference_full_min_cosine"] < 0.995:
        raise AssertionError(f"Reference Paged 与 Full Forward 超出 BF16 数值阈值: {report}")
    if include_flashinfer:
        flash = _paged_path(model, FlashInferPagedBackend, segments)
        report["flashinfer_reference_max_abs"] = max(
            float((left - right).abs().max()) for left, right in zip(flash, reference)
        )
        report["flashinfer_reference_max_relative_l2"] = max(
            float((left - right).float().norm() / right.float().norm())
            for left, right in zip(flash, reference)
        )
        report["flashinfer_reference_min_cosine"] = min(
            float(torch.nn.functional.cosine_similarity(left.float(), right.float(), dim=0))
            for left, right in zip(flash, reference)
        )
        if not all(torch.isfinite(value).all() for value in flash):
            raise AssertionError("FlashInfer 输出包含非有限值")
        if (
            report["flashinfer_reference_max_relative_l2"] > 0.05
            or report["flashinfer_reference_min_cosine"] < 0.995
        ):
            raise AssertionError(f"FlashInfer 与 Reference 超出 BF16 数值阈值: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 Neriv Paged Backend 数值与页生命周期")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--flashinfer", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.model, args.flashinfer), ensure_ascii=False))


if __name__ == "__main__":
    main()
