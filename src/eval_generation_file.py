import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from collections import defaultdict

from rule_eval import rule_score


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[警告] 跳过无效 JSONL 行: {line_no}")
    return rows


def avg(items, key):
    if not items:
        return 0.0
    return round(mean(float(x[key]) for x in items), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_file", required=True)
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.generation_file))
    if not rows:
        raise ValueError(f"生成文件为空: {args.generation_file}")

    grouped = defaultdict(list)
    details = []

    for row in rows:
        input_text = row.get("input", "")
        prediction = row.get("prediction", "")
        model_type = row.get("model_type", "unknown")

        metrics = rule_score(input_text, prediction)

        record = {
            "id": row.get("id", ""),
            "model_type": model_type,
            "input": input_text,
            "reference": row.get("reference", ""),
            "prediction": prediction,
            "coverage": metrics.get("coverage", 0.0),
            "forbidden_count": metrics.get("forbidden_count", 0),
            "repetition_ratio": metrics.get("repetition_ratio", 0.0),
            "length": metrics.get("length", 0),
            "total_score": metrics.get("total_score", 0.0),
        }

        details.append(record)
        grouped[model_type].append(record)

    summary = {}
    for model_type, items in grouped.items():
        summary[model_type] = {
            "num_samples": len(items),
            "avg_coverage": avg(items, "coverage"),
            "avg_forbidden_count": avg(items, "forbidden_count"),
            "avg_repetition_ratio": avg(items, "repetition_ratio"),
            "avg_length": avg(items, "length"),
            "avg_total_score": avg(items, "total_score"),
        }

    out_json = Path(f"outputs/eval_reports/{args.output_prefix}_eval_report.json")
    out_csv = Path(f"outputs/eval_reports/{args.output_prefix}_eval_report.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": summary,
        "details": details,
        "meta": {
            "generation_file": args.generation_file,
            "num_samples": len(rows),
        },
    }

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "system",
        "num_samples",
        "avg_coverage",
        "avg_forbidden_count",
        "avg_repetition_ratio",
        "avg_length",
        "avg_total_score",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model_type, s in summary.items():
            row = {"system": model_type}
            row.update(s)
            writer.writerow(row)

    print(f"[完成] JSON: {out_json}")
    print(f"[完成] CSV: {out_csv}")
    for model_type, s in summary.items():
        print(
            f"[{model_type}] "
            f"score={s['avg_total_score']} "
            f"coverage={s['avg_coverage']} "
            f"forbidden={s['avg_forbidden_count']} "
            f"length={s['avg_length']} "
            f"repetition={s['avg_repetition_ratio']}"
        )


if __name__ == "__main__":
    main()
