from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class TrainerState:
    """可恢复训练状态。"""

    training_started_at: str
    step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    batches_seen_in_epoch: int = 0
    collator_rng_state: Any = None

    @property
    def version(self) -> str:
        compact_time = datetime.fromisoformat(self.training_started_at).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{compact_time}-s{self.step:08d}"

    @classmethod
    def fresh(cls) -> "TrainerState":
        return cls(datetime.now(timezone.utc).replace(microsecond=0).isoformat())


class CheckpointManager:
    """原子保存模型可训练参数、优化器和全部随机状态。"""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def resolve(self, value: str | Path) -> Path:
        """解析显式 checkpoint 或 `latest`。"""

        if str(value).lower() != "latest":
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        marker = self.checkpoint_dir / "latest.json"
        if not marker.exists():
            raise FileNotFoundError("没有可恢复的 latest checkpoint")
        return self.checkpoint_dir / json.loads(marker.read_text(encoding="utf-8"))["version"]

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        state: TrainerState,
        config: dict[str, Any],
    ) -> Path:
        """保存一个由启动时间和步数组成版本号的 checkpoint。"""

        target = self.checkpoint_dir / state.version
        temporary = self.checkpoint_dir / f".{state.version}.tmp"
        trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        trainable_state = {
            name: value.detach().cpu() for name, value in model.state_dict().items() if name in trainable_names
        }
        # 温度是 buffer，不在 named_parameters 中。
        trainable_state["temperature"] = model.temperature.detach().cpu()
        nonfinite = [
            name for name, value in trainable_state.items()
            if torch.is_floating_point(value) and not torch.isfinite(value).all()
        ]
        if nonfinite:
            raise FloatingPointError(f"拒绝保存非有限 checkpoint 参数: {nonfinite[:5]}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        torch.save(trainable_state, temporary / "model.pt")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
            temporary / "runtime.pt",
        )
        (temporary / "trainer_state.json").write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (temporary / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = {
            "checkpoint_version": state.version,
            "model_contract_version": config.get("contract", {}).get("model_contract_version"),
            "contract_hash": config.get("contract", {}).get("contract_hash"),
            "backbone_revision": config.get("contract", {}).get("backbone_revision"),
            "tokenizer_revision": config.get("contract", {}).get("tokenizer_revision"),
            "dataset_manifest_hash": config.get("dataset_manifest_hash"),
            "parent_checkpoint_hash": config.get("parent_checkpoint_hash"),
            "training_stage": config.get("training_stage"),
            "training_config": config,
            "seed": config.get("seed"),
            "code_revision": config.get("code_revision"),
        }
        if config.get("training_profile"):
            manifest["training_profile"] = config["training_profile"]
        (temporary / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if target.exists():
            shutil.rmtree(target)
        temporary.replace(target)
        (self.checkpoint_dir / "latest.json").write_text(
            json.dumps({"version": state.version}, ensure_ascii=False), encoding="utf-8"
        )
        return target

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        expected_config: dict[str, Any] | None = None,
    ) -> TrainerState:
        """恢复模型、优化器、调度器、计数器和随机状态。"""

        checkpoint = self.resolve(path)
        saved_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        saved_contract = saved_config.get("contract", {}).get("contract_hash")
        current_contract = getattr(getattr(model, "contract", None), "contract_hash", None)
        if saved_contract and saved_contract != current_contract:
            raise RuntimeError("CHECKPOINT_CONTRACT_MISMATCH")
        for key, expected in (expected_config or {}).items():
            if saved_config.get(key) != expected:
                raise RuntimeError(f"CHECKPOINT_CONFIG_MISMATCH: {key}")
        model_state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        missing_trainable = trainable.intersection(missing)
        if missing_trainable or unexpected:
            raise RuntimeError(f"checkpoint 参数不匹配: missing={missing_trainable}, unexpected={unexpected}")
        runtime = torch.load(checkpoint / "runtime.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(runtime["optimizer"])
        scheduler.load_state_dict(runtime["scheduler"])
        torch.set_rng_state(runtime["torch_rng"])
        if torch.cuda.is_available() and runtime["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(runtime["cuda_rng"])
        np.random.set_state(runtime["numpy_rng"])
        random.setstate(runtime["python_rng"])
        return TrainerState(**json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8")))


class JsonlLogger:
    """只追加的结构化训练日志，天然支持断点后继续写。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()
