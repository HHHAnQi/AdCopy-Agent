import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("id", "")): r for r in rows if r.get("id")}


def shorten(text: str, max_len: int = 500) -> str:
    text = str(text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", default="outputs/eval_reports/base_generations.jsonl")
    parser.add_argument("--sft_file", default="outputs/eval_reports/sft_generations.jsonl")
    parser.add_argument("--dpo_file", default="outputs/eval_reports/dpo_generations.jsonl")
    parser.add_argument("--output_file", default="docs/comparison_samples.md")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    base_rows = to_map(read_jsonl(Path(args.base_file)))
    sft_rows = to_map(read_jsonl(Path(args.sft_file)))
    dpo_rows = to_map(read_jsonl(Path(args.dpo_file)))

    common_ids = [x for x in base_rows.keys() if x in sft_rows and x in dpo_rows]
    common_ids = common_ids[: args.num_samples]

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Base / SFT / DPO Generation Comparison\n")
    lines.append(
        "This file compares generation outputs from the base model, the SFT model, "
        "and the DPO-aligned model on the same final-test samples.\n"
    )

    for idx, sid in enumerate(common_ids, start=1):
        b = base_rows[sid]
        s = sft_rows[sid]
        d = dpo_rows[sid]

        lines.append(f"\n## Sample {idx}: `{sid}`\n")
        lines.append("### Input\n")
        lines.append(f"{shorten(b.get('input', ''))}\n")
        lines.append("### Reference\n")
        lines.append(f"{shorten(b.get('reference', ''))}\n")
        lines.append("### Base Output\n")
        lines.append(f"{shorten(b.get('prediction', ''))}\n")
        lines.append("### SFT Output\n")
        lines.append(f"{shorten(s.get('prediction', ''))}\n")
        lines.append("### DPO Output\n")
        lines.append(f"{shorten(d.get('prediction', ''))}\n")
        lines.append("---\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[完成] comparison samples saved to: {out_path}")
    print(f"[完成] samples: {len(common_ids)}")


if __name__ == "__main__":
    main()
