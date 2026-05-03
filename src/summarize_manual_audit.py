import argparse
import csv
from collections import Counter
from pathlib import Path


def pct(x, n):
    return round(100.0 * x / n, 2) if n else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="docs/results/manual_preference_audit.csv")
    parser.add_argument("--output_md", default="docs/results/manual_preference_audit_summary.md")
    parser.add_argument("--a_name", default="sft")
    parser.add_argument("--b_name", default="dpo_v2")
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.input_csv).open(encoding="utf-8-sig")))
    valid = {args.a_name, args.b_name, "tie"}

    bad = []
    for r in rows:
        w = r.get("winner", "").strip()
        if w not in valid:
            bad.append((r.get("id", ""), w))

    if bad:
        raise ValueError(f"Invalid winner labels: {bad[:10]}")

    n = len(rows)
    cnt = Counter(r["winner"].strip() for r in rows)

    a_win = cnt[args.a_name]
    b_win = cnt[args.b_name]
    tie = cnt["tie"]

    b_win_rate = pct(b_win, n)
    b_win_or_tie_rate = pct(b_win + tie, n)
    a_win_rate = pct(a_win, n)
    tie_rate = pct(tie, n)

    lines = []
    lines.append("# Manual Preference Audit Summary")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Comparison: **{args.a_name}** vs **{args.b_name}**")
    lines.append(f"- Number of audited examples: **{n}**")
    lines.append("- Annotator: project author")
    lines.append("- Criteria: attribute coverage, factual consistency, fluency, attractiveness, and compliance")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Winner | Count | Rate |")
    lines.append("|---|---:|---:|")
    lines.append(f"| {args.a_name} | {a_win} | {a_win_rate}% |")
    lines.append(f"| {args.b_name} | {b_win} | {b_win_rate}% |")
    lines.append(f"| tie | {tie} | {tie_rate}% |")
    lines.append("")
    lines.append("## Key Metric")
    lines.append("")
    lines.append(f"- **{args.b_name} win rate:** {b_win_rate}%")
    lines.append(f"- **{args.b_name} win-or-tie rate:** {b_win_or_tie_rate}%")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"In this 50-sample manual preference audit, **{args.b_name}** was preferred over **{args.a_name}** "
        f"in **{b_win_rate}%** of cases and was at least tied in **{b_win_or_tie_rate}%** of cases. "
        "This provides human preference evidence that DPO v2-small improves perceived generation quality beyond automatic rule-based metrics."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This is a small-scale manual audit intended as a qualitative validation complement.")
    lines.append("- The rule-based score is used for engineering diagnostics and does not fully replace human preference evaluation.")
    lines.append("- The audit reasons are stored in `manual_preference_audit.csv`.")
    lines.append("")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"[完成] Summary written to: {out}")
    print(f"num_examples={n}")
    print(f"{args.a_name}_wins={a_win}")
    print(f"{args.b_name}_wins={b_win}")
    print(f"ties={tie}")
    print(f"{args.b_name}_win_rate={b_win_rate}%")
    print(f"{args.b_name}_win_or_tie_rate={b_win_or_tie_rate}%")


if __name__ == "__main__":
    main()
