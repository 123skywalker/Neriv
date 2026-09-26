"""模型质量评测的整体、选择性预测与切片指标。"""

from __future__ import annotations

import torch

from .dataset import DecisionExample
from .losses import ranked_probability_score


def candidate_bucket(count: int) -> str:
    """按普通候选数分桶，不包含 Reject 槽。

    @param count 普通候选数，范围为 2 到 16。
    @return 候选数对应的固定桶标签。
    """

    if count == 2:
        return "2"
    if 3 <= count <= 5:
        return "3-5"
    if 6 <= count <= 16:
        return "6-16"
    raise ValueError("普通候选数必须位于 [2,16]")


def pad_distributions(values: list[torch.Tensor]) -> torch.Tensor:
    """把变长候选概率补零，Reject 始终放在最后一列。

    @return 可直接计算整体质量指标的二维分布。
    """

    width = max(len(value) for value in values)
    return torch.stack([
        torch.cat((value[:-1], value.new_zeros(width - len(value)), value[-1:]))
        for value in values
    ])


def _ece(confidence: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    """计算等宽分桶的 Expected Calibration Error。"""

    error = confidence.new_zeros(())
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.float().mean() * (
                confidence[selected].mean() - correct[selected].float().mean()
            ).abs()
    return float(error)


def _risk_coverage(confidence: torch.Tensor, correct: torch.Tensor) -> dict[str, object]:
    """返回风险覆盖曲线与 5% 目标风险下的最大覆盖率。"""

    order = confidence.argsort(descending=True)
    errors = (~correct[order]).float()
    positions = torch.arange(1, len(errors) + 1, device=confidence.device)
    risk = errors.cumsum(0) / positions
    coverage = positions.float() / len(errors)
    valid = coverage[risk <= 0.05]
    points = [
        {"coverage": float(coverage[index]), "risk": float(risk[index])}
        for index in torch.linspace(0, len(errors) - 1, min(20, len(errors)), device=confidence.device).long()
    ]
    return {
        "coverage_at_target_risk_0.05": float(valid.max()) if len(valid) else 0.0,
        "curve": points,
    }


def quality_metrics(predicted: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """按现有 overall 定义计算 Accuracy、NLL、Brier 与 ECE。

    @param predicted 普通候选补零且 Reject 位于最后一列的预测分布。
    @param expected 与预测同形状的目标分布。
    @return 四项质量指标。
    """

    confidence = predicted.max(-1).values
    correct = predicted.argmax(-1) == expected.argmax(-1)
    nll = -(expected * predicted.clamp_min(1e-8).log()).sum(-1).mean()
    brier = ((predicted - expected) ** 2).sum(-1).mean()
    return {
        "accuracy": float(correct.float().mean()),
        "nll": float(nll),
        "brier": float(brier),
        "ece": _ece(confidence, correct),
    }


def selective_metrics(predicted: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    """报告预测 Reject 率；样本不足 200 时不估计覆盖曲线。

    @return 预测拒答率及足量样本的风险覆盖指标。
    """

    confidence = predicted.max(-1).values
    correct = predicted.argmax(-1) == expected.argmax(-1)
    reject_rate = float((predicted.argmax(-1) == predicted.shape[-1] - 1).float().mean())
    result: dict[str, object] = {
        "reject_rate": reject_rate,
        "predicted_reject_rate": reject_rate,
    }
    if len(predicted) >= 200:
        result.update(_risk_coverage(confidence, correct))
    return result


def _score_metrics(probabilities: list[torch.Tensor], targets: list[torch.Tensor]) -> dict[str, float]:
    """按真实 K 计算 RPS；等级错误取普通候选 argmax，Reject 单独统计。"""

    rps = [ranked_probability_score(predicted[:-1], expected[:-1])
           for predicted, expected in zip(probabilities, targets)]
    distances = [abs(int(predicted[:-1].argmax()) - int(expected[:-1].argmax()))
                 for predicted, expected in zip(probabilities, targets)]
    return {
        "rps": float(torch.stack(rps).mean()),
        "ordinal_mae": sum(distances) / len(distances),
        "adjacent_error_rate": sum(distance == 1 for distance in distances) / len(distances),
        "far_error_rate": sum(distance >= 2 for distance in distances) / len(distances),
    }


def _hard_choice_metrics(examples: list[DecisionExample], probabilities: list[torch.Tensor],
                         targets: list[torch.Tensor]) -> dict[str, float]:
    """统计高置信错误、Hard 负例混淆及完整 Nested-K 组稳定性。"""

    confident_errors = 0
    hard_confusions = 0
    groups: dict[str, dict[int, str]] = {}
    for example, predicted, expected in zip(examples, probabilities, targets):
        winner = int(predicted.argmax())
        wrong = winner != int(expected.argmax())
        confident_errors += int(wrong and float(predicted[winner]) >= .8)
        hard_confusions += int(wrong and winner < len(example.candidate_ids) and
                               example.candidate_ids[winner] in example.hard_candidate_ids)
        if "/k" in example.sample_id:
            base, k = example.sample_id.rsplit("/k", 1)
            if k.isdigit():
                selected = example.candidate_ids[winner] if winner < len(example.candidate_ids) else "REJECT"
                groups.setdefault(base, {})[int(k)] = selected
    full = [values for values in groups.values() if set(values) == {2, 4, 8, 16}]
    result = {
        "confident_error_rate": confident_errors / len(examples),
        "hard_negative_top1_confusion_rate": hard_confusions / len(examples),
    }
    if full:
        result["nested_k_stability"] = sum(len(set(values.values())) == 1 for values in full) / len(full)
    return result


def slice_reports(
    examples: list[DecisionExample], probabilities: list[torch.Tensor],
    targets: list[torch.Tensor], labels: list[str],
) -> dict[str, dict[str, object]]:
    """按标签聚合质量指标；纯 Score 切片才附加有序指标。

    @param labels 与单问题样本逐一对齐的切片标签。
    @return 以切片标签为键的质量报告。
    """

    if not len(examples) == len(probabilities) == len(targets) == len(labels):
        raise ValueError("切片标签与评测样本数量不一致")
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    reports = {}
    for label, indices in groups.items():
        predicted_rows = [probabilities[index] for index in indices]
        expected_rows = [targets[index] for index in indices]
        predicted, expected = pad_distributions(predicted_rows), pad_distributions(expected_rows)
        report: dict[str, object] = {
            "samples": len(indices),
            "quality": quality_metrics(predicted, expected),
            "selective_prediction": selective_metrics(predicted, expected),
        }
        if all(examples[index].question_type == "score" for index in indices):
            report["score"] = _score_metrics(predicted_rows, expected_rows)
        if all(examples[index].candidate_policy == "nested-k-v1" for index in indices):
            report["hard_choice"] = _hard_choice_metrics(
                [examples[index] for index in indices], predicted_rows, expected_rows,
            )
        reports[label] = report
    return reports
