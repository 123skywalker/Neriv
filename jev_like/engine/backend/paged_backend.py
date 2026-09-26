from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from jev_like.engine.storage import PagedKVStore

from .protocol import PagedBatch


class QwenPagedBackend(ABC):
    """复用 Qwen 权重、只替换 Attention Kernel 的 Paged Runner。"""

    def __init__(self, backbone: torch.nn.Module, device: torch.device | str) -> None:
        self.backbone = backbone
        self._device = torch.device(device)
        self.config = backbone.config

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    def _append_and_attention(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch: PagedBatch,
        store: PagedKVStore,
        scale: float,
    ) -> torch.Tensor:
        """写入本层 K/V 并返回 packed Attention 输出。"""

    def _prepare(self, batch: PagedBatch, store: PagedKVStore) -> None:
        """在逐层计算前准备一次 Batch 级 Backend 状态。"""

    @torch.inference_mode()
    def run(self, batch: PagedBatch, store: PagedKVStore) -> tuple[torch.Tensor, ...]:
        """对 Ragged Token 一次执行完整 Qwen Backbone。"""

        token_data = batch.token_data.to(self.device, non_blocking=True)
        hidden = self.backbone.embed_tokens(token_data)
        positions = torch.cat([
            torch.arange(
                int(batch.old_seq_lens[row]),
                int(batch.old_seq_lens[row]) + int(batch.qo_indptr[row + 1] - batch.qo_indptr[row]),
                device=self.device,
            )
            for row in range(len(batch.old_seq_lens))
        ]).long()
        cos, sin = self.backbone.rotary_emb(hidden.unsqueeze(0), positions.unsqueeze(0))
        self._prepare(batch, store)
        for layer_index, layer in enumerate(self.backbone.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attention = layer.self_attn
            query = attention.q_norm(attention.q_proj(normalized).view(-1, self.config.num_attention_heads, attention.head_dim))
            key = attention.k_norm(attention.k_proj(normalized).view(-1, self.config.num_key_value_heads, attention.head_dim))
            value = attention.v_proj(normalized).view(-1, self.config.num_key_value_heads, attention.head_dim)
            rotated_query, rotated_key = apply_rotary_pos_emb(
                query.transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                cos,
                sin,
            )
            query = rotated_query.squeeze(0).transpose(0, 1).contiguous()
            key = rotated_key.squeeze(0).transpose(0, 1).contiguous()
            attended = self._append_and_attention(
                layer_index, query, key, value, batch, store, attention.scaling
            )
            hidden = residual + attention.o_proj(attended.reshape(len(hidden), -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = self.backbone.norm(hidden)
        return tuple(hidden[int(batch.qo_indptr[row + 1]) - 1] for row in range(len(batch.old_seq_lens)))


class ReferencePagedBackend(QwenPagedBackend):
    """Windows/Linux 数值测试用 PyTorch SDPA Paged Reference Backend。"""

    @staticmethod
    def _write(
        layer_cache: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch: PagedBatch,
        page_size: int,
    ) -> None:
        for row in range(len(batch.old_seq_lens)):
            token_begin = int(batch.qo_indptr[row])
            token_end = int(batch.qo_indptr[row + 1])
            page_begin = int(batch.page_indptr[row])
            page_end = int(batch.page_indptr[row + 1])
            pages = batch.page_indices[page_begin:page_end].tolist()
            old_len = int(batch.old_seq_lens[row])
            for offset, token_index in enumerate(range(token_begin, token_end)):
                position = old_len + offset
                page_id = pages[position // page_size]
                slot = position % page_size
                layer_cache[page_id, 0, slot].copy_(key[token_index])
                layer_cache[page_id, 1, slot].copy_(value[token_index])

    def _append_and_attention(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch: PagedBatch,
        store: PagedKVStore,
        scale: float,
    ) -> torch.Tensor:
        layer_cache = store.layer(layer_index)
        self._write(layer_cache, key, value, batch, store.page_size)
        outputs: list[torch.Tensor] = []
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        for row in range(len(batch.old_seq_lens)):
            q_begin, q_end = int(batch.qo_indptr[row]), int(batch.qo_indptr[row + 1])
            p_begin, p_end = int(batch.page_indptr[row]), int(batch.page_indptr[row + 1])
            pages = batch.page_indices[p_begin:p_end].long()
            total_len = int(batch.old_seq_lens[row]) + q_end - q_begin
            cached = layer_cache.index_select(0, pages).permute(1, 0, 2, 3, 4).reshape(
                2, -1, self.config.num_key_value_heads, self.config.head_dim
            )[:, :total_len]
            keys = cached[0].repeat_interleave(groups, dim=1).transpose(0, 1)
            values = cached[1].repeat_interleave(groups, dim=1).transpose(0, 1)
            queries = query[q_begin:q_end].transpose(0, 1)
            old_len = int(batch.old_seq_lens[row])
            allowed = torch.arange(total_len, device=self.device).unsqueeze(0) <= (
                old_len + torch.arange(q_end - q_begin, device=self.device).unsqueeze(1)
            )
            output = F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=allowed.unsqueeze(0), scale=scale
            )
            outputs.append(output.transpose(0, 1))
        return torch.cat(outputs)


class FlashInferPagedBackend(QwenPagedBackend):
    """Linux 正式后端：FlashInfer Batch Prefill + Paged KV。"""

    def __init__(self, backbone: torch.nn.Module, device: torch.device | str, workspace_bytes: int = 128 << 20) -> None:
        super().__init__(backbone, device)
        try:
            import flashinfer
        except ImportError as error:
            raise RuntimeError("正式 Engine 需要 Linux FlashInfer") from error
        self.flashinfer = flashinfer
        self.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=self.device)
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD")

    def _prepare(self, batch: PagedBatch, store: PagedKVStore) -> None:
        self.wrapper.plan(
            batch.qo_indptr,
            batch.page_indptr,
            batch.page_indices,
            batch.last_page_len,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            store.page_size,
            causal=True,
            pos_encoding_mode="NONE",
            sm_scale=self.config.head_dim ** -0.5,
            q_data_type=next(self.backbone.parameters()).dtype,
        )

    def _append_and_attention(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch: PagedBatch,
        store: PagedKVStore,
        scale: float,
    ) -> torch.Tensor:
        cache = store.layer(layer_index)
        self.flashinfer.append_paged_kv_cache(
            key,
            value,
            batch.append_batch_indices,
            batch.append_positions,
            cache,
            batch.page_indices,
            batch.page_indptr,
            batch.last_page_len,
            kv_layout="NHD",
        )
        return self.wrapper.run(query, cache)
