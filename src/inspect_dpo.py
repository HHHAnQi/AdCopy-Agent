import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from rule_eval import detect_forbidden_words, rule_score


def load_json_or_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {file_path}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    rows: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            print(f"[警告] 跳过无效 JSONL 行: {i}")
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def normalize_text(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def print_samples(samples: List[Dict[str, str]]) -> None:
    print("\n=== 随机样本检查 ===")
    for i, s in enumerate(samples, start=1):
        print(f"\n[样本 {i}]")
        print(f"instruction: {s['instruction']}")
        print(f"input: {s['input']}")
        print(f"chosen: {s['chosen']}")
        print(f"rejected: {s['rejected']}")
        print(f"chosen_source: {s['chosen_source']}")
        print(f"rejected_source: {s['rejected_source']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人工检查 DPO 数据质量")
    parser.add_argument(
        "--input_file",
        type=str,
        default="outputs/dpo/dpo_train_with_meta.jsonl",
        help="DPO 输入文件",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="随机打印样本数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    num_samples = max(0, args.num_samples)

    rows = load_json_or_jsonl(input_path)
    clean_rows: List[Dict[str, str]] = []

    for r in rows:
        clean_rows.append(
            {
                "instruction": normalize_text(r.get("instruction")),
                "input": normalize_text(r.get("input")),
                "chosen": normalize_text(r.get("chosen")),
                "rejected": normalize_text(r.get("rejected")),
                "chosen_source": normalize_text(r.get("chosen_source")) or "unknown",
                "rejected_source": normalize_text(r.get("rejected_source")) or "unknown",
                "chosen_score": r.get("chosen_score"),
                "rejected_score": r.get("rejected_score"),
            }
        )

    total = len(clean_rows)
    if total == 0:
        print("[信息] 输入文件无有效样本。")
        return

    random.seed(42)
    k = min(num_samples, total)
    sampled = random.sample(clean_rows, k=k) if k > 0 else []
    if sampled:
        print_samples(sampled)

    chosen_lens = [len(x["chosen"]) for x in clean_rows]
    rejected_lens = [len(x["rejected"]) for x in clean_rows]

    same_text_count = sum(1 for x in clean_rows if x["chosen"] == x["rejected"])
    chosen_forbidden_count = sum(
        1 for x in clean_rows if len(detect_forbidden_words(x["chosen"])) > 0
    )
    rejected_forbidden_count = sum(
        1 for x in clean_rows if len(detect_forbidden_words(x["rejected"])) > 0
    )

    abnormal_score_count = 0
    chosen_scores: List[float] = []
    rejected_scores: List[float] = []
    chosen_source_counter: Counter = Counter()
    rejected_source_counter: Counter = Counter()
    rejected_scores_by_source: Dict[str, List[float]] = defaultdict(list)

    for x in clean_rows:
        raw_chosen_score = x.get("chosen_score")
        raw_rejected_score = x.get("rejected_score")

        # 优先使用文件中已有 score；若不存在则回退重算，兼容旧数据
        if raw_chosen_score is None:
            chosen_score = float(rule_score(x["input"], x["chosen"])["total_score"])
        else:
            chosen_score = float(raw_chosen_score)

        if raw_rejected_score is None:
            rejected_score = float(rule_score(x["input"], x["rejected"])["total_score"])
        else:
            rejected_score = float(raw_rejected_score)

        chosen_scores.append(chosen_score)
        rejected_scores.append(rejected_score)
        chosen_source = x["chosen_source"]
        rejected_source = x["rejected_source"]
        chosen_source_counter[chosen_source] += 1
        rejected_source_counter[rejected_source] += 1
        rejected_scores_by_source[rejected_source].append(rejected_score)

        if chosen_score <= rejected_score:
            abnormal_score_count += 1

    rejected_forbidden_ratio = (
        rejected_forbidden_count / total if total > 0 else 0.0
    )

    print("\n=== DPO 数据统计 ===")
    print(f"DPO 样本总数: {total}")
    print(f"chosen 平均长度: {mean(chosen_lens):.2f}")
    print(f"rejected 平均长度: {mean(rejected_lens):.2f}")
    print(f"chosen 与 rejected 完全相同数量: {same_text_count}")
    print(f"chosen 包含禁用词数量: {chosen_forbidden_count}")
    print(f"rejected 包含禁用词数量: {rejected_forbidden_count}")
    print(f"rejected 包含禁用词比例: {rejected_forbidden_ratio:.4f}")
    print(f"chosen_score <= rejected_score 异常数量: {abnormal_score_count}")
    print(f"chosen 平均 score: {mean(chosen_scores):.4f}")
    print(f"rejected 平均 score: {mean(rejected_scores):.4f}")

    print("\nchosen_source 分布:")
    for source, cnt in sorted(chosen_source_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {source}: {cnt}")

    print("\nrejected_source 分布:")
    for source, cnt in sorted(rejected_source_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {source}: {cnt}")

    print("\n每种 rejected_source 的平均 rejected_score:")
    for source, scores in sorted(
        rejected_scores_by_source.items(), key=lambda x: (-len(x[1]), x[0])
    ):
        avg_score = mean(scores) if scores else 0.0
        print(f"  {source}: 数量={len(scores)}, 平均分={avg_score:.4f}")


if __name__ == "__main__":
    main()

