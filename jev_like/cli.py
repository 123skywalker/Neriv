from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from jev_like.engine import EngineConfig, NerivEngine
from jev_like.model.contract import DecisionInput
from jev_like.model.decision_model import load_training_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 Neriv 概率决策")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", type=Path, help="请求 JSON；省略时读取 stdin")
    parser.add_argument("--model", type=Path, help="覆盖 checkpoint 中的原始模型路径")
    parser.add_argument("--attention-backend", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--engine-backend", default="auto", choices=["auto", "flashinfer", "reference"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model, contract = load_training_checkpoint(
        args.checkpoint, args.model, attention_backend=args.attention_backend, dtype=dtype
    )
    engine = NerivEngine(
        model,
        EngineConfig(attention_backend=args.engine_backend),
        device=device,
    )
    payload = json.loads(args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read())
    values = payload if isinstance(payload, list) else [payload]
    compiled = [contract.compile(DecisionInput.from_dict(value)) for value in values]
    responses = engine.decide_batch(compiled)
    result = [asdict(response) for response in responses] if isinstance(payload, list) else asdict(responses[0])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
