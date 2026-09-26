from __future__ import annotations

import importlib.util
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .set_pointer import SetPointerHead
from .contract import ModelContract


def resolve_attention_backend(requested: str) -> str:
    """根据运行环境选择 Transformers Attention 后端。"""

    if requested not in {"auto", "flash_attention_2", "sdpa", "eager"}:
        raise ValueError(f"未知 attention backend: {requested}")
    if requested != "auto":
        return requested
    if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"


class NerivDecisionModel(nn.Module):
    """Qwen 编码器与集合指针头的最小组合。"""

    def __init__(self, backbone: nn.Module, pointer: SetPointerHead, contract: ModelContract | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.pointer = pointer
        self.contract = contract
        object.__setattr__(self, "_pointer_forward", pointer.forward)
        self.register_buffer("temperature", torch.tensor(1.0), persistent=True)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        attention_backend: str = "auto",
        dtype: torch.dtype | None = None,
        lora: bool = False,
        lora_rank: int = 16,
        max_sequence_tokens: int = 512,
    ) -> tuple["NerivDecisionModel", ModelContract]:
        """加载本地骨干，并按需注入 LoRA。

        @return: 模型与对应 tokenizer。
        """

        model_path = str(model_path)
        backend = resolve_attention_backend(attention_backend)
        load_kwargs: dict[str, Any] = {"dtype": dtype or "auto", "attn_implementation": backend}
        try:
            backbone = AutoModel.from_pretrained(model_path, **load_kwargs)
        except (ImportError, RuntimeError, ValueError) as error:
            if backend == "sdpa":
                raise
            warnings.warn(f"{backend} 加载失败，回退到 SDPA: {error}", stacklevel=2)
            load_kwargs["attn_implementation"] = "sdpa"
            backbone = AutoModel.from_pretrained(model_path, **load_kwargs)
        if lora:
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0.05,
                bias="none",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
                ],
            )
            backbone = get_peft_model(backbone, config)
        hidden_size = backbone.config.hidden_size
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        revision = cls._local_revision(Path(model_path))
        contract = ModelContract(
            tokenizer, "Qwen/Qwen3-0.6B", revision, revision, max_sequence_tokens
        )
        model = cls(backbone, SetPointerHead(hidden_size), contract)
        return model, contract

    @staticmethod
    def _local_revision(model_path: Path) -> str:
        """为本地权重与 tokenizer 全量资产计算稳定版本指纹。"""

        digest = hashlib.sha256()
        names = {
            "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
            "merges.txt", "model.safetensors.index.json",
        }
        paths = sorted(
            (path for path in model_path.iterdir() if path.name in names or path.suffix == ".safetensors"),
            key=lambda path: path.name,
        )
        if not paths:
            raise FileNotFoundError(f"模型目录缺少可指纹化资产: {model_path}")
        for path in paths:
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as stream:
                while chunk := stream.read(8 << 20):
                    digest.update(chunk)
        return f"local:{digest.hexdigest()}"

    def freeze_backbone(self) -> None:
        """冻结骨干，仅训练 Set Pointer Head。"""

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _last_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, positions]

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """编码序列并读取最后一个有效 token 的 hidden state。"""

        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        return self._last_hidden(output[0], attention_mask)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        candidate_input_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """批量编码问题与候选，并返回含 reject 的 logits。"""

        batch_size, candidate_count, sequence_length = candidate_input_ids.shape
        query = self.encode(query_input_ids, query_attention_mask)
        flat_ids = candidate_input_ids.reshape(batch_size * candidate_count, sequence_length)
        flat_mask = candidate_attention_mask.reshape(batch_size * candidate_count, sequence_length)
        candidates = self.encode(flat_ids, flat_mask).reshape(batch_size, candidate_count, -1)
        pointer_dtype = next(self.pointer.parameters()).dtype
        query = query.to(pointer_dtype)
        candidates = candidates.to(pointer_dtype)
        with torch.autocast(device_type=query.device.type, enabled=False):
            return self._pointer_forward(query, candidates, candidate_mask, self.temperature)

    def compile_pointer(self) -> None:
        """只编译形状稳定的 Set Pointer 子图，避免动态骨干反复重编译。"""

        compiled = torch.compile(self.pointer.forward, dynamic=True)
        object.__setattr__(self, "_pointer_forward", compiled)

    def save_components(self, path: str | Path) -> None:
        """保存骨干适配器/模型、Pointer Head 和温度。"""

        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output / "backbone")
        torch.save(
            {"pointer": self.pointer.state_dict(), "temperature": self.temperature.detach().cpu()},
            output / "decision_head.pt",
        )

    def load_components(self, path: str | Path) -> None:
        """加载 Pointer Head 和温度；LoRA 骨干由调用方按配置恢复。"""

        state = torch.load(Path(path) / "decision_head.pt", map_location="cpu", weights_only=True)
        self.pointer.load_state_dict(state["pointer"])
        self.temperature.copy_(state["temperature"].to(self.temperature.device))


def load_training_checkpoint(
    checkpoint: str | Path,
    model_path: str | Path | None = None,
    attention_backend: str = "auto",
    dtype: torch.dtype | None = None,
) -> tuple[NerivDecisionModel, ModelContract]:
    """从训练 checkpoint 配置恢复可推理模型。"""

    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    base_path = model_path or config["model"]
    model, contract = NerivDecisionModel.from_pretrained(
        base_path,
        attention_backend=attention_backend,
        dtype=dtype,
        lora=config.get("stage") != "head",
        lora_rank=int(config.get("lora_rank", 16)),
        max_sequence_tokens=int(config.get("max_length", 512)),
    )
    if config.get("stage") == "head":
        model.freeze_backbone()
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    expected_contract = config.get("contract", {})
    if expected_contract and expected_contract.get("contract_hash") != contract.contract_hash:
        raise RuntimeError("CHECKPOINT_CONTRACT_MISMATCH")
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trainable = trainable.intersection(missing)
    if missing_trainable or unexpected:
        raise RuntimeError(f"checkpoint 参数不匹配: missing={missing_trainable}, unexpected={unexpected}")
    return model, contract
