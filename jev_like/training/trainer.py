from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, set_seed

from jev_like.model.decision_model import NerivDecisionModel

from .checkpoint import CheckpointManager, JsonlLogger, TrainerState
from .dataset import DecisionCollator, JsonlDecisionDataset
from .losses import proper_reward, ranked_probability_score, rlcd_loss, soft_cross_entropy


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _sha256_file(path: Path) -> str:
    """流式计算大 checkpoint 指纹，不把权重文件整体读入主存。"""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_path(args: argparse.Namespace) -> Path:
    """解析显式或 latest checkpoint，供训练配置与状态使用同一来源。"""

    if args.resume != "latest":
        return Path(args.resume)
    root = Path(args.output) / "checkpoints"
    version = json.loads((root / "latest.json").read_text(encoding="utf-8"))["version"]
    return root / version


def _compile_if_supported(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    """仅编译稳定的 Pointer 子图；动态 Transformers 骨干保持 Flash SDPA eager。"""

    if not enabled:
        return model
    try:
        import triton  # noqa: F401
    except ImportError:
        print(json.dumps({"event": "compile_fallback", "reason": "Triton 不可用"}, ensure_ascii=False))
        return model
    if platform.system() == "Windows":
        print(json.dumps({"event": "compile_fallback", "reason": "Windows Triton/Inductor 非稳定默认路径"}, ensure_ascii=False))
        return model
    model.compile_pointer()
    return model


def _epoch_indices(dataset: JsonlDecisionDataset, stage: str, seed: int, epoch: int) -> list[int]:
    """按固定 seed 构造 Hardening 或 Pro 配比，其他阶段均匀排列。"""

    generator = torch.Generator().manual_seed(seed + epoch)
    if stage not in {"hardening-sft", "pro-sft"}:
        return torch.randperm(len(dataset), generator=generator).tolist()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(dataset.examples):
        if stage == "pro-sft":
            category = ("programmatic" if example.category == "programmatic" else
                        "score" if example.question_type == "score" else
                        "noul" if example.question_type == "noul" and example.dataset == "boolq" else
                        "choice" if example.dataset in {"banking77", "clinc150", "massive"} else "")
        else:
            category = example.category
        if category:
            groups[category].append(index)
    if stage == "pro-sft":
        ratios = ({"score": .55, "noul": .20, "choice": .15, "programmatic": .10}
                  if groups.get("programmatic") else {"score": .60, "noul": .25, "choice": .15})
    else:
        ratios = {"hard_choice": .45, "score": .20, "workflow": .15,
                  "native_mcq": .10, "replay": .10}
        if not groups.get("workflow"):
            ratios["hard_choice"] += .075
            ratios["replay"] += .075
            del ratios["workflow"]
    if any(not groups.get(name) for name in ratios):
        raise ValueError(f"{stage} 数据缺少必需类别")
    sampled = []
    for position, (name, fraction) in enumerate(ratios.items()):
        source = groups[name]
        count = len(dataset) - len(sampled) if position == len(ratios) - 1 else round(len(dataset) * fraction)
        while count:
            take = min(count, len(source))
            sampled.extend(source[index] for index in
                           torch.randperm(len(source), generator=generator)[:take].tolist())
            count -= take
    return [sampled[index] for index in torch.randperm(len(sampled), generator=generator).tolist()]


def train(args: argparse.Namespace) -> Path:
    """执行可恢复训练并返回最终 checkpoint 路径。"""

    set_seed(args.seed)
    resume_config = json.loads((_resume_path(args) / "config.json").read_text(encoding="utf-8")) if args.resume else None
    if args.max_length is None:
        checkpoint = Path(args.parent_checkpoint) if args.parent_checkpoint else None
        if args.resume:
            checkpoint = _resume_path(args)
        args.max_length = (int(json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
                               ["contract"]["max_sequence_tokens"]) if checkpoint else 256)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported() and not args.fp32
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device.type == "cuda" and not args.fp32 else torch.float32)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    lora = args.stage != "head"
    model, contract = NerivDecisionModel.from_pretrained(
        args.model,
        attention_backend=args.attention_backend,
        dtype=dtype,
        lora=lora,
        lora_rank=args.lora_rank,
        max_sequence_tokens=args.max_length,
    )
    if args.stage == "head":
        model.freeze_backbone()
    elif args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone, "enable_input_require_grads"):
            model.backbone.enable_input_require_grads()
    model.to(device)

    dataset = JsonlDecisionDataset(args.data)
    if not dataset:
        raise ValueError(f"训练数据为空: {args.data}")
    if args.stage == "rlcd" and any(example.split != "train" for example in dataset.examples):
        raise ValueError("RLCD 只允许 train split 更新权重")
    pro_stage = args.stage in {"pro-sft", "pro-reject-refresh", "workflow-specialist-v2"}
    if pro_stage and resume_config:
        args.effective_epochs = args.effective_epochs or resume_config.get("effective_epochs")
        args.rps_weight = resume_config.get("rps_weight") if args.rps_weight is None else args.rps_weight
        args.max_steps = resume_config["max_steps"] if args.max_steps is None else args.max_steps
    if pro_stage and any(example.split != "train" for example in dataset.examples):
        raise ValueError("Pro 权重更新只允许 train split")
    if args.stage == "pro-sft" and any(
        example.question_type == "noul" and example.dataset != "boolq" and example.category != "programmatic"
        for example in dataset.examples
    ):
        raise ValueError("Pro Noul Recovery 只允许 BoolQ 或已审计 Programmatic 数据")
    if args.stage in {"pro-sft", "workflow-specialist-v2"} and args.no_permutation:
        raise ValueError("Pro 训练必须开启候选 permutation")
    if args.stage in {"pro-sft", "workflow-specialist-v2"}:
        args.subset_ratio = args.reject_ratio = args.mismatch_ratio = 0.0
        args.effective_epochs = args.effective_epochs or (1.5 if args.stage == "pro-sft" else 2.0)
    if args.max_steps is None:
        if args.stage in {"pro-sft", "workflow-specialist-v2"}:
            args.max_steps = math.ceil(len(dataset) * args.effective_epochs /
                                       (args.batch_size * args.gradient_accumulation))
        elif args.stage == "pro-reject-refresh":
            parent_config = json.loads((Path(args.parent_checkpoint) / "config.json").read_text(encoding="utf-8"))
            args.max_steps = min(1000, max(1, round(parent_config["max_steps"] * .15)))
        else:
            args.max_steps = 1000
    if args.stage in {"pro-sft", "workflow-specialist-v2"}:
        limit = 2 if args.stage == "pro-sft" else 3
        if args.max_steps > math.ceil(len(dataset) * limit / (args.batch_size * args.gradient_accumulation)):
            raise ValueError(f"{args.stage} 超过 {limit} effective epochs")
    if args.stage == "pro-reject-refresh" and args.max_steps > 1000:
        raise ValueError("Pro Reject Refresh 不得超过 1000 updates")
    if args.rps_weight is None:
        args.rps_weight = .1 if args.stage in {"pro-sft", "workflow-specialist-v2"} else 0.0
    outlier_states: list[str] = []
    if args.stage == "pro-reject-refresh":
        args.outlier_states = args.outlier_states or Path(args.data).with_name("outlier_states.jsonl")
        with args.outlier_states.open(encoding="utf-8") as stream:
            outlier_states = [json.loads(line)["state"] for line in stream if line.strip()]
        if not outlier_states:
            raise ValueError("Pro Reject Refresh 缺少 CLINC150 OOS 训练状态")
    collator = DecisionCollator(
        contract,
        max_length=args.max_length,
        permutation=not args.no_permutation,
        subset_ratio=args.subset_ratio if args.stage not in {"head", "pro-sft", "workflow-specialist-v2"} else 0.0,
        reject_ratio=args.reject_ratio if args.stage in {"reject", "reject-refresh", "pro-reject-refresh", "rlcd"} else 0.0,
        mismatch_ratio=args.mismatch_ratio if args.stage in {"reject", "reject-refresh", "pro-reject-refresh", "rlcd"} else 0.0,
        outlier_states=outlier_states,
        outlier_ratio=args.outlier_ratio if args.stage == "pro-reject-refresh" else 0.0,
        permute_score=args.stage in {"pro-sft", "workflow-specialist-v2"},
        seed=args.seed,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, args.max_steps),
        num_training_steps=args.max_steps,
    )
    manifest_path = Path(args.data).resolve().parents[1] / "manifests" / "dataset_manifest.json"
    if pro_stage:
        if not manifest_path.is_file():
            raise FileNotFoundError("Pro 训练缺少数据 manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = {"pro-sft": "train", "pro-reject-refresh": "reject_train",
                 "workflow-specialist-v2": "specialist_train"}[args.stage]
        if (manifest.get("audit_status") != "PASS" or Path(args.data).stem != split or
                manifest.get("splits", {}).get(split, {}).get("sha256") != _sha256_file(Path(args.data))):
            raise ValueError("Pro 训练数据与已审计 manifest 不一致")
        if args.stage == "pro-reject-refresh" and (
            manifest.get("outlier_states", {}).get("sha256") != _sha256_file(args.outlier_states)
        ):
            raise ValueError("CLINC150 OOS 状态与已审计 manifest 不一致")
    dataset_manifest_hash = (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else "unavailable"
    )
    checkpoints = CheckpointManager(args.output)
    state = TrainerState.fresh()
    parent_hash = None
    if args.parent_checkpoint:
        parent = Path(args.parent_checkpoint)
        parent_config = json.loads((parent / "config.json").read_text(encoding="utf-8"))
        parent_contract = parent_config.get("contract", {}).get("contract_hash")
        if parent_contract != contract.contract_hash:
            raise RuntimeError("PARENT_CHECKPOINT_CONTRACT_MISMATCH")
        allowed_parent_stages = {
            "general-sft": {"head"},
            "hardening-sft": {"general-sft", "reject", "reject-sft"},
            "reject": {"hardening-sft"},
            "reject-refresh": {"hardening-sft"},
            "rlcd": {"reject", "reject-sft", "reject-refresh"},
            "pro-sft": {"reject-refresh"},
            "pro-reject-refresh": {"pro-sft"},
            "workflow-specialist-v2": {"pro-reject-refresh"},
        }
        parent_stage = parent_config.get("training_stage", parent_config.get("stage"))
        if parent_stage == "calibration" and parent_config.get("stage") == "reject-refresh":
            parent_stage = "reject-refresh"
        if parent_stage not in allowed_parent_stages.get(args.stage, set()):
            raise RuntimeError(f"非法 checkpoint lineage: {parent_stage} -> {args.stage}")
        if args.stage == "pro-reject-refresh":
            minimum = min(1000, max(1, math.floor(parent_config["max_steps"] * .1)))
            maximum = min(1000, max(1, math.ceil(parent_config["max_steps"] * .2)))
            if not minimum <= args.max_steps <= maximum:
                raise ValueError("Pro Reject Refresh 必须为 Pro SFT updates 的 10%–20%，且不超过 1000")
        parent_state = torch.load(parent / "model.pt", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(parent_state, strict=False)
        if unexpected or any(name.startswith("pointer.") for name in missing):
            raise RuntimeError(f"父 checkpoint 参数不兼容: missing={missing}, unexpected={unexpected}")
        if pro_stage:
            trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
            if trainable.intersection(missing):
                raise RuntimeError(f"父 checkpoint 缺少 Pro 可训练参数: {trainable.intersection(missing)}")
            model.temperature.fill_(1.0)
        parent_hash = _sha256_file(parent / "model.pt")
        if args.stage == "pro-reject-refresh":
            if not args.promotion_report:
                raise ValueError("Pro Reject Refresh 必须提供已接受的 --promotion-report")
            promotion = json.loads(Path(args.promotion_report).read_text(encoding="utf-8"))
            if promotion.get("status") != "accepted" or promotion.get("selected", {}).get("checkpoint_hash") != parent_hash:
                raise ValueError("Promotion 报告未接受当前 Pro SFT checkpoint")
    if args.resume:
        expected = {
            "stage": args.stage, "seed": args.seed, "max_length": args.max_length,
            "lora_rank": args.lora_rank, "data": str(args.data),
            "dataset_manifest_hash": dataset_manifest_hash,
        }
        if pro_stage:
            expected.update({key: getattr(args, key) for key in (
                "max_steps", "effective_epochs", "batch_size", "gradient_accumulation",
                "learning_rate", "weight_decay", "warmup_steps", "max_grad_norm",
                "rps_weight", "subset_ratio", "reject_ratio", "mismatch_ratio", "outlier_ratio",
            )})
        state = checkpoints.load(
            args.resume, model, optimizer, scheduler, expected,
        )
        resumed_config = json.loads(
            (checkpoints.resolve(args.resume) / "config.json").read_text(encoding="utf-8")
        )
        parent_hash = resumed_config.get("parent_checkpoint_hash")
        _optimizer_to(optimizer, device)
        if state.collator_rng_state is not None:
            collator.set_random_state(state.collator_rng_state)
    logger = JsonlLogger(Path(args.output) / "train_log.jsonl")
    raw_model = model
    reference_model = None
    reference_hash = None
    if args.stage == "rlcd" and args.rlcd_kl_weight:
        reference_path = Path(args.rlcd_reference_checkpoint or args.parent_checkpoint or "")
        if not (reference_path / "model.pt").is_file():
            raise ValueError("RLCD KL 需要 --rlcd-reference-checkpoint 指向 reject-sft checkpoint")
        reference_config = json.loads((reference_path / "config.json").read_text(encoding="utf-8"))
        if reference_config.get("training_stage", reference_config.get("stage")) not in {"reject", "reject-sft"}:
            raise ValueError("RLCD KL reference 必须是 reject-sft checkpoint")
        if reference_config.get("contract", {}).get("contract_hash") != contract.contract_hash:
            raise ValueError("RLCD KL reference 与当前 Model Contract 不一致")
        reference_hash = _sha256_file(reference_path / "model.pt")
        reference_model, _ = NerivDecisionModel.from_pretrained(
            args.model, attention_backend=args.attention_backend, dtype=dtype,
            lora=True, lora_rank=args.lora_rank, max_sequence_tokens=args.max_length,
        )
        reference_model.load_state_dict(torch.load(reference_path / "model.pt", map_location="cpu", weights_only=True))
        reference_model.to(device).eval().requires_grad_(False)
    training_model = _compile_if_supported(model, args.compile)
    training_model.train()
    optimizer.zero_grad(set_to_none=True)
    last_save_time = time.monotonic()
    last_log_time = time.monotonic()
    running_loss = 0.0
    running_updates = 0
    micro_step = 0
    try:
        code_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_revision = "unavailable"
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update({
        "contract": contract.manifest(),
        "dataset_manifest_hash": dataset_manifest_hash,
        "code_revision": code_revision,
        "parent_checkpoint_hash": parent_hash,
        "training_stage": args.stage,
    })
    if pro_stage:
        config["training_profile"] = "pro-v2"
    if args.stage == "rlcd":
        config["rlcd_estimator"] = "centered_gaussian_score_function_group_mean_v1"
        config["rlcd_reference_checkpoint"] = str(args.rlcd_reference_checkpoint or args.parent_checkpoint or "")
        config["rlcd_reference_checkpoint_hash"] = reference_hash

    while state.step < args.max_steps:
        if state.batches_seen_in_epoch == 0:
            state.epoch += 1
        indices = _epoch_indices(dataset, args.stage, args.seed, state.epoch)
        offset = state.batches_seen_in_epoch * args.batch_size
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=indices[offset:],
            collate_fn=collator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for batch in loader:
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits = training_model(
                    query_input_ids=batch["query_input_ids"],
                    query_attention_mask=batch["query_attention_mask"],
                    candidate_input_ids=batch["candidate_input_ids"],
                    candidate_attention_mask=batch["candidate_attention_mask"],
                    candidate_mask=batch["candidate_mask"],
                )
                supervised = soft_cross_entropy(logits, batch["targets"])
                loss = supervised
                if args.stage in {"hardening-sft", "pro-sft", "workflow-specialist-v2"} and args.rps_weight:
                    probabilities = F.softmax(logits.float(), -1)
                    score_rows = torch.where(batch["ordinal_mask"])[0]
                    if len(score_rows):
                        rps = []
                        for row in score_rows.tolist():
                            count = int(batch["candidate_mask"][row].sum())
                            order = batch["ordinal_order"][row, :count]
                            rps.append(ranked_probability_score(
                                probabilities[row, :count][order], batch["targets"][row, :count][order],
                            ))
                        loss = loss + args.rps_weight * torch.stack(rps).sum() / len(batch["targets"])
                if args.stage == "rlcd":
                    if args.rlcd_objective == "rlcd":
                        objective = rlcd_loss(
                            logits, batch["targets"], batch["ordinal_mask"],
                            valid_mask=batch["candidate_mask"], groups=args.rlcd_groups,
                            noise_std=args.rlcd_noise, spherical_weight=args.spherical_weight,
                            rps_weight=args.rps_weight,
                        )
                    elif args.rlcd_objective == "proper":
                        objective = -proper_reward(
                            F.softmax(logits.float(), -1).unsqueeze(0), batch["targets"],
                            batch["ordinal_mask"], batch["candidate_mask"],
                            args.spherical_weight, args.rps_weight,
                        ).mean()
                    else:
                        objective = 0.0
                    loss = objective + args.rlcd_lambda_sft * supervised
                    if reference_model is not None:
                        with torch.no_grad():
                            reference_logits = reference_model(
                                query_input_ids=batch["query_input_ids"],
                                query_attention_mask=batch["query_attention_mask"],
                                candidate_input_ids=batch["candidate_input_ids"],
                                candidate_attention_mask=batch["candidate_attention_mask"],
                                candidate_mask=batch["candidate_mask"],
                            )
                        loss = loss + args.rlcd_kl_weight * F.kl_div(
                            F.log_softmax(logits.float(), -1),
                            F.softmax(reference_logits.float(), -1), reduction="batchmean",
                        )
                loss = loss / args.gradient_accumulation
            loss.backward()
            running_loss += float(loss.detach()) * args.gradient_accumulation
            state.samples_seen += len(batch["targets"])
            state.batches_seen_in_epoch += 1
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state.step += 1
            running_updates += 1
            now = time.monotonic()
            if state.step % args.log_every == 0 or now - last_log_time >= args.log_every_seconds:
                payload = {
                    "event": "train",
                    "training_started_at": state.training_started_at,
                    "checkpoint_version": state.version,
                    "step": state.step,
                    "epoch": state.epoch,
                    "samples_seen": state.samples_seen,
                    "loss": running_loss / max(running_updates, 1),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                logger.log(payload)
                print(json.dumps(payload, ensure_ascii=False), flush=True)
                running_loss = 0.0
                running_updates = 0
                last_log_time = now
            if state.step % args.save_every == 0 or now - last_save_time >= args.save_every_seconds:
                state.collator_rng_state = collator.get_random_state()
                path = checkpoints.save(raw_model, optimizer, scheduler, state, config)
                logger.log({"event": "checkpoint", "checkpoint_version": state.version, "path": str(path)})
                last_save_time = now
            if state.step >= args.max_steps:
                break
        else:
            state.batches_seen_in_epoch = 0
            continue
        break
    state.collator_rng_state = collator.get_random_state()
    final_path = checkpoints.save(raw_model, optimizer, scheduler, state, config)
    logger.log({"event": "complete", "checkpoint_version": state.version, "path": str(final_path)})
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 Neriv Set Pointer 模型")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--stage", choices=["head", "general-sft", "hardening-sft", "reject", "reject-refresh", "pro-sft", "pro-reject-refresh", "workflow-specialist-v2", "rlcd"], default="head"
    )
    parser.add_argument("--resume", help="checkpoint 路径或 latest")
    parser.add_argument("--parent-checkpoint", help="跨阶段只继承模型权重的父 checkpoint")
    parser.add_argument("--promotion-report", type=Path, help="Pro SFT Promotion 的已接受报告")
    parser.add_argument("--attention-backend", choices=["auto", "flash_attention_2", "sdpa", "eager"], default="sdpa")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, help="默认继承父 checkpoint，否则为 256")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--effective-epochs", type=float)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--reject-ratio", type=float, default=0.15)
    parser.add_argument("--subset-ratio", type=float, default=0.25)
    parser.add_argument("--mismatch-ratio", type=float, default=0.05)
    parser.add_argument("--outlier-states", type=Path)
    parser.add_argument("--outlier-ratio", type=float, default=0.05)
    parser.add_argument("--rlcd-groups", type=int, default=4)
    parser.add_argument("--rlcd-objective", choices=["softce", "proper", "rlcd"], default="rlcd")
    parser.add_argument("--rlcd-noise", type=float, default=0.1)
    parser.add_argument("--rlcd-lambda-sft", type=float, default=1.0)
    parser.add_argument("--rlcd-kl-weight", type=float, default=0.0)
    parser.add_argument("--rlcd-reference-checkpoint", type=Path)
    parser.add_argument("--spherical-weight", type=float, default=0.0)
    parser.add_argument("--rps-weight", type=float)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-every-seconds", type=float, default=60.0)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--save-every-seconds", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-permutation", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.reject_ratio < 1.0:
        raise SystemExit("--reject-ratio 必须位于 [0, 1)")
    if args.resume and args.parent_checkpoint:
        raise SystemExit("--resume 与 --parent-checkpoint 不能同时使用")
    if args.stage != "head" and not args.resume and not args.parent_checkpoint:
        raise SystemExit(f"{args.stage} 必须通过 --parent-checkpoint 继承上一阶段，或使用 --resume")
    if args.stage == "rlcd" and (args.rlcd_lambda_sft <= 0 or args.rlcd_kl_weight < 0):
        raise SystemExit("RLCD 要求 lambda_sft > 0 且 kl_weight >= 0")
    if args.rps_weight is not None and args.rps_weight < 0:
        raise SystemExit("--rps-weight 不能为负数")
    if args.batch_size <= 0 or args.gradient_accumulation <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        raise SystemExit("batch-size、gradient-accumulation 与 max-steps 必须为正")
    if args.stage == "pro-reject-refresh" and not 0 < args.outlier_ratio < 1:
        raise SystemExit("Pro Reject Refresh 的 outlier-ratio 必须位于 (0,1)")
    if args.effective_epochs is not None and not 0 < args.effective_epochs <= (3 if args.stage == "workflow-specialist-v2" else 2):
        raise SystemExit("--effective-epochs 超过阶段上限")
    path = train(args)
    print(json.dumps({"final_checkpoint": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
