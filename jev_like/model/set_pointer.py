from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SetPointerHead(nn.Module):
    """无候选位置编码的单层集合注意力与指针头。"""

    def __init__(self, hidden_size: int, pointer_size: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        self.query_norm = nn.RMSNorm(hidden_size)
        self.candidate_norm = nn.LayerNorm(hidden_size)
        self.set_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.set_ffn_norm = nn.LayerNorm(hidden_size)
        self.set_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.query_projection = nn.Linear(hidden_size, pointer_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, pointer_size, bias=False)
        self.null_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.null_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        temperature: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """计算普通候选和末尾 reject 候选的 logits。

        @param query: `[B,D]` 的问题表示。
        @param candidates: `[B,K,D]` 的独立候选表示。
        @param candidate_mask: `[B,K]`，真值表示有效普通候选。
        @param temperature: 校准温度，必须为正。
        @return: `[B,K+1]` logits，最后一列为 reject。
        """

        if query.ndim != 2 or candidates.ndim != 3 or candidate_mask.shape != candidates.shape[:2]:
            raise ValueError("SetPointerHead 输入形状不合法")
        null = self.null_projection(query) + self.null_embedding
        values = torch.cat((candidates, null.unsqueeze(1)), dim=1)
        valid_mask = torch.cat(
            (candidate_mask.bool(), torch.ones((query.shape[0], 1), device=query.device, dtype=torch.bool)), dim=1
        )
        normalized = self.candidate_norm(values)
        attended, _ = self.set_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        values = values + attended
        values = values + self.set_ffn(self.set_ffn_norm(values))
        q = F.normalize(self.query_projection(self.query_norm(query)), dim=-1)
        keys = F.normalize(self.key_projection(values), dim=-1)
        logits = self.log_scale.exp().clamp(max=100.0) * torch.einsum("bd,bkd->bk", q, keys)
        logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
        temperature_tensor = torch.as_tensor(temperature, device=logits.device, dtype=logits.dtype)
        if torch.any(temperature_tensor <= 0):
            raise ValueError("temperature 必须大于 0")
        return logits / temperature_tensor

