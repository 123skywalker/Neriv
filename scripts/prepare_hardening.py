"""下载并审计 Hardening 数据，使用冻结 checkpoint 构造 Nested-K。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset

from jev_like.data.io import read_jsonl, write_jsonl
from jev_like.data.normalize import BANKING77_LABELS
from jev_like.data.prepare import prepare_dataset
from jev_like.data.schema import Candidate, CanonicalSample, Question, Target
from jev_like.model.decision_model import load_training_checkpoint


SOURCES = {
    "banking77": ("mteb/banking77", None, "18072d2685ea682290f7b8924d94c62acc19c0b2", "mit"),
    "massive": ("AmazonScience/massive", "en-US", "ed58ac423a2f4121720918bf5301577edce4ffd3", "cc-by-4.0"),
    "sciq": ("allenai/sciq", None, "2c94ad3e1aafab77146f384e23536f97a4849815", "cc-by-nc-3.0"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453", "cc-by-sa-4.0"),
    "openbookqa": ("allenai/openbookqa", "main", "388097ea7776314e93a529163e0fea805b8a6454", "apache-2.0"),
}
CLINC_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
SPLITS = {"train": "train", "validation": "dev", "test": "locked_test"}
PROMPT = "What is the user's intent?"


def _sha256(path: Path) -> str:
    """流式计算文件指纹。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(name: str, split: str, root: Path, token: str | None) -> list[dict]:
    """优先读取已下载 JSONL；首次下载后以原始行落盘。"""

    path = root / "raw" / name / f"{split}.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    repo, config, revision, _ = SOURCES[name]
    if name == "massive":
        url = (f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
               f"en-US/{split}/0000.parquet")
        dataset = load_dataset("parquet", data_files={split: url}, split=split, streaming=True)
    else:
        dataset = load_dataset(repo, config, split=split, revision=revision, streaming=True, token=token)
    rows = list(dataset)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return rows


def _clinc_rows(root: Path) -> dict[str, list[dict]]:
    """读取官方固定 revision 的 CLINC150，保留原生 split 与 OOS。"""

    path = root / "raw" / "clinc150" / "data_full.json"
    if not path.is_file():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/clinc/oos-eval/{CLINC_REVISION}/data/data_full.json"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        path.write_bytes(data)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {split: [{"text": text, "label": label} for text, label in raw[key]]
            for key, split in (("train", "train"), ("val", "validation"), ("test", "test"))}


def _choice(name: str, split: str, index: int, row: dict, labels: list[str]) -> dict:
    """把大标签空间行整理成未选候选的中间记录。"""

    if name == "banking77":
        state, label, source_id = row["text"], labels[int(row["label"])], str(index)
        if row.get("label_text", label).lower().rstrip("?") != label:
            raise ValueError("Banking77 标签编号与候选全集不一致")
    elif name == "massive":
        state, label, source_id = row["utt"], labels[int(row["intent"])], str(row["id"])
    else:
        state, label, source_id = row["text"], row["label"], str(index)
    sample_id = f"{name}/{split}/{source_id}"
    target_split = SPLITS[split]
    if split == "train":
        bucket = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) % 100
        if bucket < 10:
            target_split = "calibration"
        elif name == "banking77" and bucket < 20:
            target_split = "dev"
    return {"id": sample_id, "state": state.strip(), "gold": label,
            "dataset": name, "split": target_split}


def _mcq(name: str, split: str, index: int, row: dict) -> CanonicalSample | None:
    """保留题干、原始干扰项及正确答案，不把答案字段写入 state。"""

    if name == "sciq":
        answers = [row["correct_answer"], *(row[f"distractor{i}"] for i in (1, 2, 3))]
        source_id, state = str(index), row["question"]
        gold = 0
    else:
        choices = row["choices"]
        answers = choices["text"]
        source_id = str(row["id"])
        state = row.get("question", row.get("question_stem"))
        if row["answerKey"] not in choices["label"]:
            return None
        gold = choices["label"].index(row["answerKey"])
    if not 2 <= len(answers) <= 16 or len(set(answers)) != len(answers):
        return None
    return CanonicalSample(
        f"{name}/{split}/{source_id}", state.strip(), [Question(
            "q0", "choice", "Which answer is correct?",
            [Candidate(f"c{i}", text.strip()) for i, text in enumerate(answers)],
            Target([float(i == gold) for i in range(len(answers))]),
        )], {"dataset": name, "split": SPLITS[split], "group_id": f"{name}/{source_id}",
              "category": "native_mcq", "candidate_policy": "fixed-candidates-v1"},
    )


def _clean_splits(records: list[CanonicalSample]) -> tuple[list[CanonicalSample], dict[str, int]]:
    """优先保留评测 split，移除跨 split 状态或 group 重合。"""

    order = {"locked_test": 0, "dev": 1, "calibration": 2, "train": 3}
    seen_states: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    seen_ids: set[str] = set()
    retained = []
    dropped: dict[str, int] = defaultdict(int)
    for record in sorted(records, key=lambda item: order[item.meta["split"]]):
        record.validate()
        split = record.meta["split"]
        state_hash = hashlib.sha256(" ".join(record.state.lower().split()).encode()).hexdigest()
        group = str(record.meta["group_id"])
        if (record.sample_id in seen_ids or seen_states.get(state_hash, split) != split or
                seen_groups.get(group, split) != split):
            dropped[split] += 1
            continue
        seen_ids.add(record.sample_id)
        seen_states[state_hash] = split
        seen_groups[group] = split
        retained.append(record)
    return retained, dict(dropped)


@torch.inference_mode()
def _mine_pools(
    rows: list[dict], universes: dict[str, list[str]], checkpoint: Path, model_path: Path,
) -> tuple[dict[str, dict[str, list[str]]], str]:
    """用冻结 checkpoint 的编码器余弦分数挖各标签的全量候选 Hard Pool。"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model, contract = load_training_checkpoint(checkpoint, model_path, dtype=dtype)
    model.to(device).eval()
    representatives: dict[tuple[str, str], str] = {}
    for row in rows:
        if row["split"] == "train":
            representatives.setdefault((row["dataset"], row["gold"]), row["state"])
    pools: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for (dataset, gold), state in representatives.items():
        query = list(contract.state_tokens(state)) + list(contract.question_tokens(PROMPT))
        input_ids = torch.tensor([query], device=device)
        query_vector = torch.nn.functional.normalize(
            model.encode(input_ids, torch.ones_like(input_ids)).float(), dim=-1,
        )
        scored = []
        candidates = [(candidate, query + list(contract.candidate_tokens(candidate.replace("_", " "))))
                      for candidate in universes[dataset]]
        for offset in range(0, len(candidates), 16):
            chunk = [(name, tokens) for name, tokens in candidates[offset:offset + 16]
                     if len(tokens) <= contract.max_sequence_tokens]
            if not chunk:
                continue
            width = max(len(tokens) for _, tokens in chunk)
            ids = torch.full((len(chunk), width), contract.tokenizer.pad_token_id,
                             dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for row, (_, tokens) in enumerate(chunk):
                ids[row, :len(tokens)] = torch.tensor(tokens, device=device)
                mask[row, :len(tokens)] = 1
            vectors = torch.nn.functional.normalize(model.encode(ids, mask).float(), dim=-1)
            scores = (vectors * query_vector).sum(-1).tolist()
            scored.extend(zip(scores, (name for name, _ in chunk)))
        pools[dataset][gold] = [name for _, name in sorted(scored, reverse=True) if name != gold]
    return dict(pools), _sha256(checkpoint / "model.pt")


def _nested(row: dict, hard: list[str], universe: list[str], seed: int) -> list[CanonicalSample]:
    """构造包含关系严格成立的 2/4/8/16 候选集。"""

    gold = row["gold"]
    hard = [name for name in hard if name != gold][:8]
    random_pool = [name for name in universe if name != gold and name not in hard]
    generator = random.Random(hashlib.sha256(f"{seed}:{row['id']}:0".encode()).hexdigest())
    generator.shuffle(random_pool)
    selected = [gold]
    output = []
    for count, hard_count in ((2, 1), (4, 2), (8, 4), (16, 8)):
        wanted = [gold, *hard[:hard_count], *random_pool[:count - hard_count - 1]]
        for name in wanted:
            if name not in selected:
                selected.append(name)
        if len(selected) < count:
            raise ValueError(f"{row['dataset']} 候选全集不足 {count} 个")
        options = selected[:count]
        output.append(CanonicalSample(
            f"{row['id']}/k{count}", row["state"], [Question(
                "intent", "choice", PROMPT,
                [Candidate(name, name.replace("_", " ")) for name in options],
                Target([float(name == gold) for name in options]),
            )], {"dataset": row["dataset"], "split": row["split"], "group_id": row["id"],
                  "category": "hard_choice", "candidate_policy": "nested-k-v1", "k": count,
                  "hard_candidate_ids": [name for name in options if name in hard]},
        ))
    return output


def prepare(root: Path, checkpoint: Path, model_path: Path, seed: int = 42, token: str | None = None) -> dict:
    """下载来源、挖冻结 Hard Pool、审计后写出可训练 IR 与 manifest。"""

    raw = {name: {split: _rows(name, split, root, token)
                  for split in (("train", "test") if name == "banking77" else SPLITS)}
           for name in SOURCES}
    raw["clinc150"] = _clinc_rows(root)
    massive_url = (f"https://huggingface.co/datasets/AmazonScience/massive/resolve/"
                   f"{SOURCES['massive'][2]}/en-US/train/0000.parquet")
    massive_features = load_dataset("parquet", data_files={"train": massive_url},
                                    split="train", streaming=True).features
    universes = {
        "banking77": BANKING77_LABELS,
        "massive": massive_features["intent"].names,
        "clinc150": sorted({row["label"] for rows in raw["clinc150"].values()
                            for row in rows if row["label"] != "oos"}),
    }
    choice_rows = []
    for name in universes:
        for split, rows in raw[name].items():
            if name == "clinc150":
                rows = [row for row in rows if row["label"] != "oos"]
            choice_rows.extend(_choice(name, split, i, row, universes[name])
                               for i, row in enumerate(rows))
    universe_hash = hashlib.sha256(json.dumps(universes, sort_keys=True).encode()).hexdigest()
    miner_hash = _sha256(checkpoint / "model.pt")
    pool_path = root / "raw" / "hard_pool.json"
    cached = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.is_file() else {}
    if cached.get("mining_checkpoint_hash") == miner_hash and cached.get("candidate_universe_hash") == universe_hash:
        pools = cached["pools"]
    else:
        pools, miner_hash = _mine_pools(choice_rows, universes, checkpoint, model_path)
        pool_path.write_text(json.dumps({"mining_checkpoint_hash": miner_hash,
                                         "candidate_universe_hash": universe_hash,
                                         "pools": pools}, ensure_ascii=False), encoding="utf-8")
    records = [_mcq(name, split, i, row) for name in ("sciq", "arc_challenge", "openbookqa")
               for split, rows in raw[name].items() for i, row in enumerate(rows)]
    valid_records = [record for record in records if record is not None]
    for row in choice_rows:
        valid_records.extend(_nested(row, pools[row["dataset"]][row["gold"]],
                                     universes[row["dataset"]], seed))
    for name in ("sst5", "yelp_review_full", "amazon_reviews_multi"):
        score_records, _ = prepare_dataset(name, root, 20_000, token)
        for record in score_records:
            split = record.meta["split"]
            if split == "train" and int(hashlib.sha256(record.sample_id.encode()).hexdigest()[:8], 16) % 100 < 10:
                split = "calibration"
            group_prefix = name if name == "amazon_reviews_multi" else record.meta["split"] + "/" + name
            valid_records.append(CanonicalSample(record.sample_id, record.state, record.questions,
                                                  {**record.meta, "split": split, "category": "score",
                                                   "group_id": f"{group_prefix}/{record.meta.get('group_id', record.sample_id)}"}))
    legacy_root = root.parent
    legacy_paths = {
        "train": legacy_root / "normalized" / "general_train.jsonl",
        "dev": legacy_root / "splits" / "dev.jsonl",
        "calibration": legacy_root / "splits" / "calibration.jsonl",
        "locked_test": legacy_root / "splits" / "locked_test.jsonl",
    }
    for split, path in legacy_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"缺少现有 Score / Replay 数据: {path}")
        for record in read_jsonl(path):
            name = record.meta.get("dataset")
            if name in {"boolq", "ag_news", "multi_nli", "trec", "dbpedia_14", "imdb"}:
                category = "replay"
            else:
                continue
            group_prefix = name if name == "multi_nli" else record.meta["split"] + "/" + name
            valid_records.append(CanonicalSample(record.sample_id, record.state, record.questions,
                                                  {**record.meta, "split": split, "category": category,
                                                   "group_id": f"{group_prefix}/{record.meta.get('group_id', record.sample_id)}"}))
    valid_records, dropped = _clean_splits(valid_records)
    ratio = {"hard_choice": 0.525, "score": 0.20, "native_mcq": 0.10, "replay": 0.175}
    counts = {}
    for split in ("train", "dev", "calibration", "locked_test"):
        selected = [record for record in valid_records if record.meta["split"] == split]
        count, digest = write_jsonl(root / "normalized" / f"{split}.jsonl", selected)
        counts[split] = {"records": count, "sha256": digest}
    manifest = {
        "manifest_version": "neriv-hardening-v1", "model_contract_version": "neriv-contract-v1",
        "normalizer_version": "hardening-v1", "candidate_policy": "nested-k-v1",
        "seed": seed, "mining_checkpoint_hash": miner_hash,
        "mining_method": "frozen_encoder_cosine_by_label_representative",
        "candidate_universe_hash": universe_hash,
        "hard_pool_hash": hashlib.sha256(json.dumps(pools, sort_keys=True).encode()).hexdigest(),
        "mix_policy": {"ratios": ratio, "sampling": "deterministic_shuffled_cycles",
                       "programmatic": "fallback_to_hard_choice_and_replay"},
        "sources": {name: {"repo": values[0], "revision": values[2], "license": values[3]}
                    for name, values in SOURCES.items()},
        "clinc150": {"repo": "clinc/oos-eval", "revision": CLINC_REVISION, "license": "cc-by-3.0"},
        "splits": counts, "dropped_cross_split": dropped,
        "audit_status": "PASS",
    }
    path = root / "manifests" / "dataset_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """解析输入并执行数据准备；令牌只从环境读取。"""

    import os

    parser = argparse.ArgumentParser(description="准备 Neriv Hardening 训练数据")
    parser.add_argument("--root", type=Path, default=Path("data/hardening"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare(args.root, args.checkpoint, args.model, args.seed, os.getenv("HF_TOKEN"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
