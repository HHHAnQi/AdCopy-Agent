import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return " ".join(str(x).replace("\n", " ").replace("\r", " ").split()).strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def combined_score(c: Dict[str, Any]) -> float:
    rule = c.get("rule_metrics", {})
    judge = c.get("judge_scores", {})

    rule_total = float(rule.get("total_score", 0.0))
    coverage = float(rule.get("coverage", 0.0))
    forbidden = float(rule.get("forbidden_count", 0.0))
    repetition = float(rule.get("repetition_ratio", 0.0))
    judge_total = float(judge.get("total", 0.0))

    # judge_total: 5-25，归一到 0-5
    judge_norm = judge_total / 5.0

    score = (
        0.45 * rule_total
        + 0.35 * judge_norm
        + 0.20 * coverage * 5.0
        - 0.50 * forbidden
        - 1.00 * repetition
    )
    return score


def valid_text(text: str) -> bool:
    text = normalize_text(text)
    if len(text) < 20:
        return False
    if len(text) > 220:
        return False
    return True


def make_pairs_for_row(
    row: Dict[str, Any],
    min_score_gap: float,
    max_pairs_per_prompt: int,
) -> List[Dict[str, Any]]:
    candidates = row.get("judged_candidates", [])
    cleaned = []

    for c in candidates:
        text = normalize_text(c.get("text"))
        if not valid_text(text):
            continue
        item = dict(c)
        item["combined_score"] = combined_score(item)
        cleaned.append(item)

    if len(cleaned) < 2:
        return []

    cleaned = sorted(cleaned, key=lambda x: x["combined_score"], reverse=True)

    pairs = []
    best = cleaned[0]

    for neg in cleaned[1:]:
        gap = best["combined_score"] - neg["combined_score"]
        if gap < min_score_gap:
            continue
        best_rule = float(best.get("rule_metrics", {}).get("total_score", 0.0))
        neg_rule = float(neg.get("rule_metrics", {}).get("total_score", 0.0))
        
        best_judge = float(best.get("judge_scores", {}).get("total", 0.0))
        neg_judge = float(neg.get("judge_scores", {}).get("total", 0.0))
        
        if best_rule <= neg_rule:
            continue
        
        if best_judge <= neg_judge:
            continue
        
        if best_judge - neg_judge < 2:
            continue

        pair = {
            "id": f"{row.get('id')}_pair_{len(pairs)+1}",
            "instruction": normalize_text(row.get("instruction")),
            "input": normalize_text(row.get("input")),
            "chosen": normalize_text(best.get("text")),
            "rejected": normalize_text(neg.get("text")),
            "meta": {
                "source": "dpo_v2_model_generated_judge",
                "prompt_id": row.get("id"),
                "chosen_candidate_id": best.get("candidate_id"),
                "rejected_candidate_id": neg.get("candidate_id"),
                "chosen_combined_score": best["combined_score"],
                "rejected_combined_score": neg["combined_score"],
                "score_gap": gap,
                "chosen_rule_metrics": best.get("rule_metrics", {}),
                "rejected_rule_metrics": neg.get("rule_metrics", {}),
                "chosen_judge_scores": best.get("judge_scores", {}),
                "rejected_judge_scores": neg.get("judge_scores", {}),
            },
        }
        pairs.append(pair)

        if len(pairs) >= max_pairs_per_prompt:
            break

    return pairs


def split_train_val(rows: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(seed)
    rows = list(rows)
    random.shuffle(rows)
    n_val = int(len(rows) * val_ratio)
    val = rows[:n_val]
    train = rows[n_val:]
    return train, val


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="outputs/v2/dpo_v2_judged_candidates_100.jsonl")
    parser.add_argument("--train_output", default="data/processed/dpo_v2_train.jsonl")
    parser.add_argument("--val_output", default="data/processed/dpo_v2_val.jsonl")
    parser.add_argument("--stats_output", default="outputs/v2/dpo_v2_pair_stats.json")
    parser.add_argument("--min_score_gap", type=float, default=0.3)
    parser.add_argument("--max_pairs_per_prompt", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_jsonl(Path(args.input_file))

    all_pairs = []
    prompts_with_pairs = 0

    for row in rows:
        pairs = make_pairs_for_row(
            row=row,
            min_score_gap=args.min_score_gap,
            max_pairs_per_prompt=args.max_pairs_per_prompt,
        )
        if pairs:
            prompts_with_pairs += 1
            all_pairs.extend(pairs)

    train, val = split_train_val(all_pairs, args.val_ratio, args.seed)

    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.val_output), val)

    stats = {
        "input_prompts": len(rows),
        "prompts_with_pairs": prompts_with_pairs,
        "total_pairs": len(all_pairs),
        "train_pairs": len(train),
        "val_pairs": len(val),
        "min_score_gap": args.min_score_gap,
        "max_pairs_per_prompt": args.max_pairs_per_prompt,
    }

    Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_output).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[完成] train: {args.train_output}")
    print(f"[完成] val: {args.val_output}")
    print(f"[完成] stats: {args.stats_output}")


if __name__ == "__main__":
    main()
