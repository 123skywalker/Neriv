from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jev_like.model.decision_model import load_training_checkpoint

from .dataset import DecisionCollator, DecisionExample, JsonlDecisionDataset
from .metrics import candidate_bucket, pad_distributions, quality_metrics, selective_metrics, slice_reports


def _data_hash(path: Path) -> str:
    """记录冻结评测包指纹，供同包 Promotion 对比。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    """在指定只读评测包上生成模型质量、切片与稳健性报告。"""

    if args.output.resolve() == args.data.resolve():
        raise ValueError("质量报告不得覆盖评测数据")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model, contract = load_training_checkpoint(
        args.checkpoint, args.model, attention_backend=args.attention_backend, dtype=dtype
    )
    model.to(device).eval()
    dataset = JsonlDecisionDataset(args.data)
    collator = DecisionCollator(
        contract, contract.max_sequence_tokens, permutation=False, subset_ratio=0.0
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        tensors = {name: batch[name].to(device) for name in (
            "query_input_ids", "query_attention_mask", "candidate_input_ids",
            "candidate_attention_mask", "candidate_mask",
        )}
        logits = model(
            tensors["query_input_ids"], tensors["query_attention_mask"],
            tensors["candidate_input_ids"], tensors["candidate_attention_mask"],
            tensors["candidate_mask"],
        )
        batch_probabilities = torch.softmax(logits.float(), -1).cpu()
        batch_targets = batch["targets"].float()
        counts = batch["candidate_mask"].sum(-1).tolist()
        for row, count in enumerate(counts):
            probabilities.append(torch.cat((batch_probabilities[row, :count], batch_probabilities[row, -1:])))
            targets.append(torch.cat((batch_targets[row, :count], batch_targets[row, -1:])))
    if not probabilities:
        raise ValueError("评测数据为空")
    predicted, expected = pad_distributions(probabilities), pad_distributions(targets)
    permutation_drift = _permutation_drift(model, contract, dataset.examples[: args.robustness_samples], device)
    report: dict[str, object] = {
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "samples": len(dataset),
        "quality": quality_metrics(predicted, expected),
        "slices": {
            "by_question_type": slice_reports(
                dataset.examples, probabilities, targets, [item.question_type for item in dataset.examples],
            ),
            "by_dataset": slice_reports(
                dataset.examples, probabilities, targets, [item.dataset or "unknown" for item in dataset.examples],
            ),
            "by_k": slice_reports(
                dataset.examples, probabilities, targets,
                [candidate_bucket(len(item.candidates)) for item in dataset.examples],
            ),
            "by_dataset_x_k": slice_reports(
                dataset.examples, probabilities, targets,
                [f"{item.dataset or 'unknown'}|{candidate_bucket(len(item.candidates))}"
                 for item in dataset.examples],
            ),
            "by_type_x_k": slice_reports(
                dataset.examples, probabilities, targets,
                [f"{item.question_type}|{candidate_bucket(len(item.candidates))}"
                 for item in dataset.examples],
            ),
        },
        "robustness": {
            "permutation_drift_kl": permutation_drift,
            "contract_hash": contract.contract_hash,
        },
        "selective_prediction": selective_metrics(predicted, expected),
        "eval": {"path": "decision_model", "batch_size": args.batch_size,
                 "data_sha256": _data_hash(args.data)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


@torch.inference_mode()
def _permutation_drift(model, contract, examples: list[DecisionExample], device: torch.device) -> float:
    """反转候选顺序并计算复原后的平均 KL。"""

    if not examples:
        return math.nan
    collator = DecisionCollator(contract, contract.max_sequence_tokens, False, 0.0)
    def infer(batch):
        values = {name: value.to(device) for name, value in batch.items()}
        return torch.softmax(model(
            values["query_input_ids"], values["query_attention_mask"],
            values["candidate_input_ids"], values["candidate_attention_mask"],
            values["candidate_mask"],
        ).float(), -1).cpu()

    divergences = []
    for item in examples:
        reversed_item = replace(
            item, candidate_ids=list(reversed(item.candidate_ids)),
            candidates=list(reversed(item.candidates)), probabilities=list(reversed(item.probabilities)),
        )
        first, second = infer(collator([item])), infer(collator([reversed_item]))
        restored = torch.cat((second[:, :-1].flip(1), second[:, -1:]), dim=1)
        divergences.append((
            first * (first.clamp_min(1e-8).log() - restored.clamp_min(1e-8).log())
        ).sum())
    return float(torch.stack(divergences).mean())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 Neriv 模型质量评测报告")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--variant", default="Final V1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--robustness-samples", type=int, default=64)
    parser.add_argument(
        "--attention-backend", default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    report = evaluate(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
