import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def load_json_or_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        return []

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
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def normalize_text(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def build_sft_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    output_lengths: List[int] = []
    missing_instruction = 0
    missing_input = 0
    missing_output = 0

    for r in rows:
        instruction = normalize_text(r.get("instruction"))
        input_text = normalize_text(r.get("input"))
        output = normalize_text(r.get("output"))

        if not instruction:
            missing_instruction += 1
        if not input_text:
            missing_input += 1
        if not output:
            missing_output += 1

        output_lengths.append(len(output))

    return {
        "output_avg_length": round(safe_mean([float(x) for x in output_lengths]), 2),
        "output_min_length": min(output_lengths) if output_lengths else 0,
        "output_max_length": max(output_lengths) if output_lengths else 0,
        "missing_instruction_count": missing_instruction,
        "missing_input_count": missing_input,
        "missing_output_count": missing_output,
    }


def build_dpo_stats(
    dpo_train_rows: List[Dict[str, Any]],
    dpo_val_rows: List[Dict[str, Any]],
    dpo_meta_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    all_train_val = dpo_train_rows + dpo_val_rows
    chosen_lens = [len(normalize_text(x.get("chosen"))) for x in all_train_val]
    rejected_lens = [len(normalize_text(x.get("rejected"))) for x in all_train_val]

    chosen_source_dist: Dict[str, int] = {}
    rejected_source_dist: Dict[str, int] = {}
    chosen_scores: List[float] = []
    rejected_scores: List[float] = []
    abnormal_count = 0

    for r in dpo_meta_rows:
        chosen_source = normalize_text(r.get("chosen_source")) or "unknown"
        rejected_source = normalize_text(r.get("rejected_source")) or "unknown"
        chosen_source_dist[chosen_source] = chosen_source_dist.get(chosen_source, 0) + 1
        rejected_source_dist[rejected_source] = rejected_source_dist.get(rejected_source, 0) + 1

        c_score_raw = r.get("chosen_score")
        r_score_raw = r.get("rejected_score")
        try:
            c_score = float(c_score_raw)
            chosen_scores.append(c_score)
        except (TypeError, ValueError):
            c_score = None
        try:
            rr_score = float(r_score_raw)
            rejected_scores.append(rr_score)
        except (TypeError, ValueError):
            rr_score = None

        if c_score is not None and rr_score is not None and c_score <= rr_score:
            abnormal_count += 1

    return {
        "chosen_avg_length": round(safe_mean([float(x) for x in chosen_lens]), 2),
        "rejected_avg_length": round(safe_mean([float(x) for x in rejected_lens]), 2),
        "chosen_source_distribution": dict(
            sorted(chosen_source_dist.items(), key=lambda x: (-x[1], x[0]))
        ),
        "rejected_source_distribution": dict(
            sorted(rejected_source_dist.items(), key=lambda x: (-x[1], x[0]))
        ),
        "chosen_score_avg": round(safe_mean(chosen_scores), 4),
        "rejected_score_avg": round(safe_mean(rejected_scores), 4),
        "chosen_score_le_rejected_score_count": abnormal_count,
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, report: Dict[str, Any]) -> None:
    counts = report["file_counts"]
    sft = report["sft_stats"]
    dpo = report["dpo_stats"]

    lines = [
        "# Dataset Report",
        "",
        "## 文件样本数",
        "",
    ]
    for k, v in counts.items():
        lines.append(f"- `{k}`: {v}")

    lines += [
        "",
        "## SFT 统计",
        "",
        f"- output 平均长度: {sft['output_avg_length']}",
        f"- output 最小长度: {sft['output_min_length']}",
        f"- output 最大长度: {sft['output_max_length']}",
        f"- instruction 字段缺失数量: {sft['missing_instruction_count']}",
        f"- input 字段缺失数量: {sft['missing_input_count']}",
        f"- output 字段缺失数量: {sft['missing_output_count']}",
        "",
        "## DPO 统计",
        "",
        f"- chosen 平均长度: {dpo['chosen_avg_length']}",
        f"- rejected 平均长度: {dpo['rejected_avg_length']}",
        f"- chosen_score 平均值: {dpo['chosen_score_avg']}",
        f"- rejected_score 平均值: {dpo['rejected_score_avg']}",
        f"- chosen_score <= rejected_score 数量: {dpo['chosen_score_le_rejected_score_count']}",
        "",
        "### chosen_source 分布",
        "",
    ]
    for source, cnt in dpo["chosen_source_distribution"].items():
        lines.append(f"- {source}: {cnt}")

    lines += [
        "",
        "### rejected_source 分布",
        "",
    ]
    for source, cnt in dpo["rejected_source_distribution"].items():
        lines.append(f"- {source}: {cnt}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    file_map = {
        "data/processed/sft_train.jsonl": project_root / "data/processed/sft_train.jsonl",
        "data/processed/sft_val.jsonl": project_root / "data/processed/sft_val.jsonl",
        "data/processed/dpo_prompt_pool.jsonl": project_root / "data/processed/dpo_prompt_pool.jsonl",
        "data/processed/dpo_train.jsonl": project_root / "data/processed/dpo_train.jsonl",
        "data/processed/dpo_val.jsonl": project_root / "data/processed/dpo_val.jsonl",
        "data/processed/dev_eval.jsonl": project_root / "data/processed/dev_eval.jsonl",
        "data/processed/final_test.jsonl": project_root / "data/processed/final_test.jsonl",
        "outputs/dpo/dpo_pairs_with_meta.jsonl": project_root / "outputs/dpo/dpo_pairs_with_meta.jsonl",
    }

    loaded: Dict[str, List[Dict[str, Any]]] = {
        k: load_json_or_jsonl(p) for k, p in file_map.items()
    }
    counts = {k: len(v) for k, v in loaded.items()}

    sft_rows = loaded["data/processed/sft_train.jsonl"] + loaded["data/processed/sft_val.jsonl"]
    sft_stats = build_sft_stats(sft_rows)

    dpo_stats = build_dpo_stats(
        dpo_train_rows=loaded["data/processed/dpo_train.jsonl"],
        dpo_val_rows=loaded["data/processed/dpo_val.jsonl"],
        dpo_meta_rows=loaded["outputs/dpo/dpo_pairs_with_meta.jsonl"],
    )

    report = {
        "file_counts": counts,
        "sft_stats": sft_stats,
        "dpo_stats": dpo_stats,
    }

    json_out = project_root / "outputs/eval_reports/dataset_report.json"
    md_out = project_root / "outputs/eval_reports/dataset_report.md"
    write_json(json_out, report)
    write_md(md_out, report)

    print("=== Dataset Report ===")
    for k, v in counts.items():
        print(f"- {k}: {v}")
    print("\nSFT:")
    print(
        f"- output avg/min/max: {sft_stats['output_avg_length']}/"
        f"{sft_stats['output_min_length']}/{sft_stats['output_max_length']}"
    )
    print(
        f"- missing instruction/input/output: "
        f"{sft_stats['missing_instruction_count']}/"
        f"{sft_stats['missing_input_count']}/"
        f"{sft_stats['missing_output_count']}"
    )
    print("\nDPO:")
    print(
        f"- chosen avg len: {dpo_stats['chosen_avg_length']}, "
        f"rejected avg len: {dpo_stats['rejected_avg_length']}"
    )
    print(
        f"- chosen_score avg: {dpo_stats['chosen_score_avg']}, "
        f"rejected_score avg: {dpo_stats['rejected_score_avg']}"
    )
    print(
        f"- chosen_score <= rejected_score: "
        f"{dpo_stats['chosen_score_le_rejected_score_count']}"
    )
    print(f"\n报告已保存: {json_out}")
    print(f"报告已保存: {md_out}")


if __name__ == "__main__":
    main()

