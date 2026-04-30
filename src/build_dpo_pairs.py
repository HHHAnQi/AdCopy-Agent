import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from rule_eval import rule_score

NEGATIVE_SOURCES = [
    "exaggerated",
    "generic",
    "weak",
]


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


def write_jsonl(file_path: Path, rows: List[Dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def score_candidates(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    input_text = normalize_text(item.get("input"))
    candidates = item.get("candidates", [])
    scored: List[Dict[str, Any]] = []
    if not isinstance(candidates, list):
        return scored

    for c in candidates:
        if not isinstance(c, dict):
            continue
        text = normalize_text(c.get("text"))
        source = normalize_text(c.get("source")) or "unknown"
        if not text:
            continue
        score_info = rule_score(input_text, text)
        scored.append(
            {
                "source": source,
                "text": text,
                "score": score_info,
            }
        )
    return scored


def pick_chosen(scored: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not scored:
        return None

    def chosen_qualified(x: Dict[str, Any]) -> bool:
        score = x.get("score", {})
        return (
            int(score.get("forbidden_count", 0)) == 0
            and int(score.get("conflict_count", 0)) == 0
            and float(score.get("length_score", 0.0)) >= 0.5
        )

    # 1) 优先 gold 且无禁用词、无冲突、长度合格
    gold_safe = [
        x
        for x in scored
        if x.get("source") == "gold"
        and chosen_qualified(x)
    ]
    if gold_safe:
        return max(gold_safe, key=lambda x: x["score"].get("total_score", 0.0))

    # 2) 若无合格 gold，选择总分最高的非 generic / 非 weak 候选
    safe = [
        x
        for x in scored
        if chosen_qualified(x) and x.get("source") not in {"generic", "weak"}
    ]
    if safe:
        return max(safe, key=lambda x: x["score"].get("total_score", 0.0))

    # 兜底：仍优先避免 generic/weak
    fallback = [x for x in scored if x.get("source") not in {"generic", "weak"}]
    if fallback:
        return max(fallback, key=lambda x: x["score"].get("total_score", 0.0))
    return max(scored, key=lambda x: x["score"].get("total_score", 0.0))


def rejected_has_clear_issue(candidate: Dict[str, Any]) -> bool:
    score = candidate.get("score", {})
    forbidden_count = int(score.get("forbidden_count", 0))
    conflict_count = int(score.get("conflict_count", 0))
    coverage = float(score.get("coverage", 0.0))
    length_score = float(score.get("length_score", 0.0))
    repetition_ratio = float(score.get("repetition_ratio", 0.0))
    return (
        forbidden_count > 0
        or conflict_count > 0
        or coverage < 0.55
        or length_score < 0.45
        or repetition_ratio > 0.2
    )


def build_multi_negative_pairs(
    scored: List[Dict[str, Any]],
    chosen: Dict[str, Any],
    max_pairs_per_prompt: int = 3,
) -> List[Dict[str, Any]]:
    chosen_text = normalize_text(chosen.get("text"))
    chosen_score = float(chosen.get("score", {}).get("total_score", 0.0))
    chosen_source = normalize_text(chosen.get("source")) or "unknown"

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for c in scored:
        source = normalize_text(c.get("source")) or "unknown"
        by_source.setdefault(source, []).append(c)

    # 仅允许 rejected 来自 exaggerated / weak / generic
    ordered_sources = NEGATIVE_SOURCES[:]

    pairs: List[Dict[str, Any]] = []
    used_texts = {chosen_text}
    for source in ordered_sources:
        if len(pairs) >= max_pairs_per_prompt:
            break
        source_candidates = by_source.get(source, [])
        if not source_candidates:
            continue
        source_candidates = sorted(
            source_candidates, key=lambda x: float(x["score"].get("total_score", 0.0))
        )

        selected: Optional[Dict[str, Any]] = None
        for c in source_candidates:
            rejected_text = normalize_text(c.get("text"))
            rejected_score = float(c.get("score", {}).get("total_score", 0.0))
            score_gap_ok = (chosen_score - rejected_score) >= 0.3
            clear_issue = rejected_has_clear_issue(c)
            if not rejected_text or rejected_text in used_texts:
                continue
            if score_gap_ok or clear_issue:
                selected = c
                break

        if not selected:
            continue

        rejected_text = normalize_text(selected.get("text"))
        rejected_score = float(selected.get("score", {}).get("total_score", 0.0))
        rejected_source = normalize_text(selected.get("source")) or "unknown"
        used_texts.add(rejected_text)
        pairs.append(
            {
                "chosen": chosen_text,
                "rejected": rejected_text,
                "chosen_source": chosen_source,
                "rejected_source": rejected_source,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "chosen_forbidden_count": int(chosen.get("score", {}).get("forbidden_count", 0)),
            }
        )
    return pairs


def strict_filter_pairs(raw_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    reason_counter: Counter = Counter()
    filtered: List[Dict[str, Any]] = []
    abnormal_score_count = 0

    for p in raw_pairs:
        chosen = normalize_text(p.get("chosen"))
        rejected = normalize_text(p.get("rejected"))
        chosen_score = float(p.get("chosen_score", 0.0))
        rejected_score = float(p.get("rejected_score", 0.0))
        chosen_source = normalize_text(p.get("chosen_source"))
        chosen_forbidden_count = int(p.get("chosen_forbidden_count", 0))

        if chosen_score <= rejected_score:
            reason_counter["chosen_score<=rejected_score"] += 1
            abnormal_score_count += 1
            continue
        if chosen_score - rejected_score < 0.5:
            reason_counter["score_gap<0.5"] += 1
            continue
        if chosen_forbidden_count > 0:
            reason_counter["chosen_has_forbidden"] += 1
            continue
        if chosen == rejected:
            reason_counter["chosen_equals_rejected"] += 1
            continue
        if chosen_source.startswith("generic") or chosen_source.startswith("weak"):
            reason_counter["chosen_source_generic_or_weak"] += 1
            continue

        filtered.append(p)

    return {
        "filtered_pairs": filtered,
        "reason_counter": reason_counter,
        "abnormal_score_count": abnormal_score_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从候选文案构造 DPO chosen/rejected 数据")
    parser.add_argument(
        "--input_file",
        type=str,
        default="outputs/candidates/candidates.jsonl",
        help="候选输入文件",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="data/processed/dpo_train.jsonl",
        help="DPO 训练集输出路径",
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="data/processed/dpo_val.jsonl",
        help="DPO 验证集输出路径",
    )
    parser.add_argument(
        "--scored_output",
        type=str,
        default="outputs/dpo/dpo_scored_candidates.jsonl",
        help="候选打分输出路径",
    )
    parser.add_argument(
        "--meta_output",
        type=str,
        default="outputs/dpo/dpo_pairs_with_meta.jsonl",
        help="带 source/score 的分析文件输出路径",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="验证集比例，默认 0.2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    scored_output = Path(args.scored_output)
    meta_output = Path(args.meta_output)
    val_ratio = min(max(args.val_ratio, 0.0), 0.9)

    rows = load_json_or_jsonl(input_path)

    raw_dpo_pairs: List[Dict[str, Any]] = []
    scored_rows: List[Dict[str, Any]] = []
    chosen_scores: List[float] = []
    rejected_scores: List[float] = []

    for item in rows:
        instruction = normalize_text(item.get("instruction"))
        input_text = normalize_text(item.get("input"))
        item_id = normalize_text(item.get("id"))
        reference = normalize_text(item.get("reference"))

        scored = score_candidates(item)
        chosen = pick_chosen(scored)

        # 记录打分明细，便于分析
        scored_rows.append(
            {
                "id": item_id,
                "instruction": instruction,
                "input": input_text,
                "reference": reference,
                "scored_candidates": scored,
                "selected_chosen": chosen,
            }
        )

        if not chosen:
            continue

        prompt_pairs = build_multi_negative_pairs(scored=scored, chosen=chosen, max_pairs_per_prompt=3)
        for p in prompt_pairs:
            raw_dpo_pairs.append(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "chosen": p["chosen"],
                    "rejected": p["rejected"],
                    "chosen_source": p["chosen_source"],
                    "rejected_source": p["rejected_source"],
                    "chosen_score": p["chosen_score"],
                    "rejected_score": p["rejected_score"],
                }
            )

    strict_result = strict_filter_pairs(raw_dpo_pairs)
    dpo_pairs = strict_result["filtered_pairs"]
    reason_counter = strict_result["reason_counter"]
    abnormal_score_count_filtered_out = int(strict_result["abnormal_score_count"])

    for p in dpo_pairs:
        chosen_scores.append(float(p["chosen_score"]))
        rejected_scores.append(float(p["rejected_score"]))

    random.seed(42)
    random.shuffle(dpo_pairs)

    val_size = int(len(dpo_pairs) * val_ratio)
    dpo_val_full = dpo_pairs[:val_size]
    dpo_train_full = dpo_pairs[val_size:]

    # 训练文件保持兼容格式，避免影响 LLaMA-Factory 读取
    dpo_train = [
        {
            "instruction": x["instruction"],
            "input": x["input"],
            "chosen": x["chosen"],
            "rejected": x["rejected"],
        }
        for x in dpo_train_full
    ]
    dpo_val = [
        {
            "instruction": x["instruction"],
            "input": x["input"],
            "chosen": x["chosen"],
            "rejected": x["rejected"],
        }
        for x in dpo_val_full
    ]

    write_jsonl(train_output, dpo_train)
    write_jsonl(val_output, dpo_val)
    meta_rows = [
        {
            "instruction": x["instruction"],
            "input": x["input"],
            "chosen": x["chosen"],
            "rejected": x["rejected"],
            "chosen_source": x["chosen_source"],
            "rejected_source": x["rejected_source"],
            "chosen_score": x["chosen_score"],
            "rejected_score": x["rejected_score"],
        }
        for x in dpo_pairs
    ]

    write_jsonl(meta_output, meta_rows)
    write_jsonl(scored_output, scored_rows)

    avg_chosen = mean(chosen_scores) if chosen_scores else 0.0
    avg_rejected = mean(rejected_scores) if rejected_scores else 0.0

    chosen_source_counter = Counter(x["chosen_source"] for x in dpo_pairs)
    rejected_source_counter = Counter(x["rejected_source"] for x in dpo_pairs)
    abnormal_score_count_after_filter = sum(
        1 for x in dpo_pairs if float(x["chosen_score"]) <= float(x["rejected_score"])
    )

    print(f"[完成] 原始候选 pair 数: {len(raw_dpo_pairs)}")
    print(f"[完成] 过滤后 pair 数: {len(dpo_pairs)}")
    print(f"[完成] dpo_train: {len(dpo_train)}")
    print(f"[完成] dpo_val: {len(dpo_val)}")
    print("[完成] 被过滤原因统计:")
    for reason, cnt in sorted(reason_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {reason}: {cnt}")
    print("[完成] chosen_source 分布:")
    for source, cnt in sorted(chosen_source_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {source}: {cnt}")
    print("[完成] rejected_source 分布:")
    for source, cnt in sorted(rejected_source_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {source}: {cnt}")
    print(f"[完成] chosen_score<=rejected_score（过滤掉）: {abnormal_score_count_filtered_out}")
    print(f"[完成] chosen_score <= rejected_score 异常数量: {abnormal_score_count_after_filter}")
    print(f"[完成] 平均 chosen_score: {avg_chosen:.4f}")
    print(f"[完成] 平均 rejected_score: {avg_rejected:.4f}")
    print(f"[完成] meta 输出: {meta_output}")
    print(f"[完成] scored 输出: {scored_output}")


if __name__ == "__main__":
    main()

