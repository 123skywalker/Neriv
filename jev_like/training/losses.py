from __future__ import annotations

import torch
from torch.nn import functional as F


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算软标签交叉熵。

    @param logits 含拒绝槽的模型输出。
    @param targets 与输出对齐的目标分布。
    @return 批量平均损失。
    """

    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    return -(targets * torch.where(targets > 0, log_probabilities, 0)).sum(dim=-1).mean()


def ranked_probability_score(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算有序普通候选的 Ranked Probability Score，不包含拒绝槽。"""

    return (probabilities.cumsum(-1)[..., :-1] - targets.cumsum(-1)[..., :-1]).square().sum(-1)


def proper_reward(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    ordinal_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    spherical_weight: float = 0.0,
    rps_weight: float = 0.0,
) -> torch.Tensor:
    """按样本计算 Log、Spherical 与可选的有序 RPS 奖励。"""

    expanded = targets.unsqueeze(0).expand_as(probabilities)
    reward = (expanded * probabilities.clamp_min(1e-8).log()).sum(-1)
    if spherical_weight:
        reward = reward + spherical_weight * (expanded * probabilities).sum(-1) / probabilities.norm(dim=-1).clamp_min(1e-8)
    if rps_weight:
        for row in torch.where(ordinal_mask & targets[:, -1].eq(0))[0].tolist():
            count = int(valid_mask[row].sum()) if valid_mask is not None else targets.shape[-1] - 1
            if count > 1:
                reward[:, row] -= rps_weight * ranked_probability_score(
                    probabilities[:, row, :count], expanded[:, row, :count]
                )
    return reward


def rlcd_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ordinal_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    groups: int = 4,
    noise_std: float = 0.1,
    spherical_weight: float = 0.0,
    rps_weight: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """以高斯扰动的 score-function 估计 RLCD 梯度，组均值作基线。

    @param valid_mask 普通候选的有效槽掩码；拒绝槽始终有效。
    @return 不含监督锚点及参考 KL 的探索损失。
    """

    if groups < 2 or noise_std <= 0:
        raise ValueError("RLCD 要求 groups >= 2 且 noise_std > 0")
    mask = torch.ones_like(logits, dtype=torch.bool) if valid_mask is None else torch.cat(
        (valid_mask, torch.ones_like(valid_mask[:, :1])), dim=-1
    )
    noise = torch.randn((groups, *logits.shape), device=logits.device, dtype=torch.float32, generator=generator)
    noise = (noise - noise.mean(dim=0, keepdim=True)) * noise_std * mask.unsqueeze(0)
    center = logits.float().masked_fill(~mask, 0)
    sampled_logits = center.detach().unsqueeze(0) + noise
    probabilities = F.softmax(sampled_logits.masked_fill(~mask.unsqueeze(0), float("-inf")), dim=-1)
    rewards = proper_reward(probabilities, targets.float(), ordinal_mask, valid_mask, spherical_weight, rps_weight)
    advantages = (rewards - rewards.mean(dim=0, keepdim=True)).detach()
    log_density = -((sampled_logits - center.unsqueeze(0)).square() * mask.unsqueeze(0)).sum(-1)
    log_density = log_density / (2 * noise_std**2)
    return -(advantages * log_density).mean()
