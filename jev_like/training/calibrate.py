from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from jev_like.model.decision_model import load_training_checkpoint

from .dataset import DecisionCollator, JsonlDecisionDataset


def calibrate(args: argparse.Namespace) -> float:
    """在独立 calibration set 上只拟合全局正温度。"""

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model, contract = load_training_checkpoint(
        args.checkpoint, args.model, attention_backend=args.attention_backend, dtype=dtype
    )
    model.to(device).eval()
    dataset = JsonlDecisionDataset(args.data)
    if args.max_length != contract.max_sequence_tokens:
        raise ValueError("Calibration max_length 必须与 checkpoint Model Contract 一致")
    collator = DecisionCollator(contract, args.max_length, permutation=False, subset_ratio=0.0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        model.temperature.fill_(1.0)
        for batch in loader:
            tensors = {name: value.to(device) for name, value in batch.items()}
            logits = model(
                tensors["query_input_ids"],
                tensors["query_attention_mask"],
                tensors["candidate_input_ids"],
                tensors["candidate_attention_mask"],
                tensors["candidate_mask"],
            )
            captured.append((logits.float().cpu(), tensors["targets"].float().cpu()))
    # 必须在 inference_mode 外 clone，LBFGS 才能只对温度反传。
    batches = [(logits.clone(), targets.clone()) for logits, targets in captured]
    del captured
    if not batches:
        raise ValueError("calibration 数据为空")
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=args.max_iterations)

    def nll() -> torch.Tensor:
        """用当前温度计算校准集平均 NLL。"""

        temperature = log_temperature.exp().clamp(0.05, 20.0)
        losses = [
            -(targets * F.log_softmax(logits / temperature, dim=-1)).sum()
            for logits, targets in batches
        ]
        return torch.stack(losses).sum() / len(dataset)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll()
        loss.backward()
        return loss

    with torch.no_grad():
        before = float(nll())
    optimizer.step(closure)
    temperature = float(log_temperature.exp().clamp(0.05, 20.0).detach())
    with torch.no_grad():
        after = float(nll())

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.checkpoint)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["temperature"] = temperature
    config["training_stage"] = "calibration"
    config["parent_checkpoint_hash"] = hashlib.sha256((source / "model.pt").read_bytes()).hexdigest()
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    state = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    state["temperature"] = torch.tensor(temperature)
    torch.save(state, output / "model.pt")
    (output / "calibration.json").write_text(
        json.dumps(
            {"temperature": temperature, "nll_before": before, "nll_after": after, "samples": len(dataset)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    source_manifest = json.loads(
        (source / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    source_manifest.update({
        "checkpoint_version": f"{source_manifest['checkpoint_version']}-calibrated",
        "parent_checkpoint_hash": config["parent_checkpoint_hash"],
        "training_stage": "calibration",
        "training_config": config,
        "temperature": temperature,
    })
    (output / "checkpoint_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return temperature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="拟合 Neriv 全局温度")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--attention-backend", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    temperature = calibrate(args)
    print(json.dumps({"temperature": temperature, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
