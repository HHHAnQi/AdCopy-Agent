import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from rule_eval import rule_score


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


def score_one(input_text: str, text: str) -> Dict[str, Any]:
    m = rule_score(input_text, text)
    return {
        "coverage": float(m.get("coverage", 0.0)),
        "forbidden_count": int(m.get("forbidden_count", 0)),
        "repetition_ratio": float(m.get("repetition_ratio", 0.0)),
        "length": int(m.get("length", 0)),
        "total_score": float(m.get("total_score", 0.0)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="outputs/v2/dpo_v2_candidates_100.jsonl")
    parser.add_argument("--output_file", default="outputs/v2/dpo_v2_scored_candidates_100.jsonl")
    parser.add_argument("--min_length", type=int, default=30)
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--max_forbidden", type=int, default=0)
    parser.add_argument("--max_repetition", type=float, default=0.12)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_jsonl(Path(args.input_file))

    output_rows = []
    total_candidates = 0
    kept_candidates = 0

    for row in rows:
        input_text = normalize_text(row.get("input"))
        scored = []

        for cid, text in enumerate(row.get("candidates", []), start=1):
            text = normalize_text(text)
            if not text:
                continue

            total_candidates += 1
            metrics = score_one(input_text, text)

            keep = (
                metrics["length"] >= args.min_length
                and metrics["length"] <= args.max_length
                and metrics["forbidden_count"] <= args.max_forbidden
                and metrics["repetition_ratio"] <= args.max_repetition
            )

            if keep:
                kept_candidates += 1

            scored.append(
                {
                    "candidate_id": f"{row.get('id')}_cand_{cid}",
                    "text": text,
                    "rule_metrics": metrics,
                    "keep_by_rule": keep,
                }
            )

        scored = sorted(
            scored,
            key=lambda x: (
                x["keep_by_rule"],
                x["rule_metrics"]["total_score"],
                x["rule_metrics"]["coverage"],
                -x["rule_metrics"]["forbidden_count"],
            ),
            reverse=True,
        )

        out = dict(row)
        out["scored_candidates"] = scored
        output_rows.append(out)

    write_jsonl(Path(args.output_file), output_rows)

    print(f"[完成] 输入样本数: {len(rows)}")
    print(f"[完成] 总候选数: {total_candidates}")
    print(f"[完成] rule 保留候选数: {kept_candidates}")
    print(f"[完成] 输出: {args.output_file}")


if __name__ == "__main__":
    main()
