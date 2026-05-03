import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_id(rows: List[dict]) -> Dict[str, dict]:
    return {str(r.get("id")): r for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_file", required=True)
    parser.add_argument("--b_file", required=True)
    parser.add_argument("--a_name", default="sft")
    parser.add_argument("--b_name", default="dpo_v2")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--md_output", default="docs/results/manual_preference_audit_sheet.md")
    parser.add_argument("--csv_output", default="docs/results/manual_preference_audit.csv")
    args = parser.parse_args()

    a_rows = read_jsonl(Path(args.a_file))
    b_rows = read_jsonl(Path(args.b_file))

    a_map = index_by_id(a_rows)
    b_map = index_by_id(b_rows)

    common_ids = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not common_ids:
        raise ValueError("No overlapping ids between the two generation files.")

    random.seed(args.seed)
    sampled_ids = random.sample(common_ids, min(args.num_samples, len(common_ids)))

    Path(args.md_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv_output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "input",
            "reference",
            f"{args.a_name}_prediction",
            f"{args.b_name}_prediction",
            "winner",
            "reason",
        ])

        for sid in sampled_ids:
            a = a_map[sid]
            b = b_map[sid]
            writer.writerow([
                sid,
                a.get("input", ""),
                a.get("reference", ""),
                a.get("prediction", ""),
                b.get("prediction", ""),
                "",
                "",
            ])

    lines = []
    lines.append("# Manual Preference Audit Sheet")
    lines.append("")
    lines.append(f"Comparison: **{args.a_name}** vs **{args.b_name}**")
    lines.append("")
    lines.append(f"Number of samples: **{len(sampled_ids)}**")
    lines.append("")
    lines.append("## Annotation Rule")
    lines.append("")
    lines.append("For each sample, choose one winner:")
    lines.append("")
    lines.append(f"- `{args.a_name}`: output A is better")
    lines.append(f"- `{args.b_name}`: output B is better")
    lines.append("- `tie`: both are similar")
    lines.append("")
    lines.append("Judging criteria:")
    lines.append("")
    lines.append("1. Covers provided product attributes")
    lines.append("2. Does not fabricate unsupported details")
    lines.append("3. Natural and fluent Chinese expression")
    lines.append("4. Attractive as e-commerce ad copy")
    lines.append("5. Avoids exaggerated or forbidden claims")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, sid in enumerate(sampled_ids, start=1):
        a = a_map[sid]
        b = b_map[sid]

        lines.append(f"## Sample {i}: `{sid}`")
        lines.append("")
        lines.append("### Input")
        lines.append("")
        lines.append("```text")
        lines.append(str(a.get("input", "")))
        lines.append("```")
        lines.append("")
        lines.append("### Reference")
        lines.append("")
        lines.append("```text")
        lines.append(str(a.get("reference", "")))
        lines.append("```")
        lines.append("")
        lines.append(f"### A: {args.a_name}")
        lines.append("")
        lines.append("```text")
        lines.append(str(a.get("prediction", "")))
        lines.append("```")
        lines.append("")
        lines.append(f"### B: {args.b_name}")
        lines.append("")
        lines.append("```text")
        lines.append(str(b.get("prediction", "")))
        lines.append("```")
        lines.append("")
        lines.append("### Manual Annotation")
        lines.append("")
        lines.append(f"Winner: `[ ] {args.a_name}  [ ] {args.b_name}  [ ] tie`")
        lines.append("")
        lines.append("Reason:")
        lines.append("")
        lines.append("---")
        lines.append("")

    Path(args.md_output).write_text("\n".join(lines), encoding="utf-8")

    print(f"[完成] Markdown audit sheet: {args.md_output}")
    print(f"[完成] CSV audit template: {args.csv_output}")
    print(f"[信息] sampled examples: {len(sampled_ids)}")


if __name__ == "__main__":
    main()
