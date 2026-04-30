import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from rule_eval import rule_score


SYSTEMS = ["base", "sft", "dpo"]


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


def extract_values(input_text: str) -> List[str]:
    text = normalize_text(input_text)
    if "商品属性：" in text:
        text = text.split("商品属性：", 1)[1]
    text = text.rstrip("。.")
    values: List[str] = []
    for seg in text.split("；"):
        seg = seg.strip()
        if "=" not in seg:
            continue
        _, value = seg.split("=", 1)
        value = value.strip()
        if value:
            values.append(value)
    return values


def make_base_text(reference: str, input_text: str) -> str:
    text = normalize_text(reference)
    values = extract_values(input_text)
    for v in values[::2]:
        if len(v) >= 2:
            text = text.replace(v, "")
    text = text.strip("，。； ")
    if len(text) < 18:
        text = "这款商品整体风格简洁，适合日常穿搭，舒适且百搭。"
    if not text.endswith("。"):
        text += "。"
    return text


def make_sft_text(reference: str, input_text: str) -> str:
    # 以 reference 为主，只做轻微清理，模拟更稳健输出
    text = normalize_text(reference)
    if not text:
        vals = extract_values(input_text)
        hint = "、".join(vals[:2]) if vals else "商品属性"
        text = f"这款单品围绕{hint}进行设计，风格自然，日常搭配轻松省心。"
    if not text.endswith("。"):
        text += "。"
    return text


def make_dpo_text(reference: str, input_text: str) -> str:
    # 偏好对齐后模拟：尽量覆盖属性且避免违规词
    values = extract_values(input_text)
    text = normalize_text(reference)
    if not text:
        joined = "、".join(values[:3]) if values else "核心卖点"
        text = f"这款商品围绕{joined}打造，穿着体验舒适，风格自然耐看，适合多场景搭配。"

    # 若覆盖不足，补一段属性提示语，提升 coverage
    missing = [v for v in values if v and v not in text]
    if missing:
        addon = "、".join(missing[:3])
        text = f"{text} 同时结合{addon}等设计特点，整体更具辨识度。"
    if not text.endswith("。"):
        text += "。"
    return text


def build_mock_generations(test_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    random.seed(42)
    outputs: List[Dict[str, Any]] = []
    for i, row in enumerate(test_rows, start=1):
        rid = normalize_text(row.get("id")) or f"test_{i:06d}"
        instruction = normalize_text(row.get("instruction"))
        input_text = normalize_text(row.get("input"))
        reference = normalize_text(row.get("reference") or row.get("output"))

        outputs.append(
            {
                "id": rid,
                "instruction": instruction,
                "input": input_text,
                "reference": reference,
                "generations": {
                    "base": make_base_text(reference, input_text),
                    "sft": make_sft_text(reference, input_text),
                    "dpo": make_dpo_text(reference, input_text),
                },
            }
        )
    return outputs


def load_file_mode_generations(generation_file: Path) -> List[Dict[str, Any]]:
    """
    file 模式输入约定（JSON 或 JSONL）每条至少包含：
    {
      "id": "...",
      "instruction": "...",
      "input": "...",
      "reference": "...",
      "generations": {"base":"...", "sft":"...", "dpo":"..."}
    }
    """
    rows = load_json_or_jsonl(generation_file)
    normalized: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        gens = row.get("generations", {})
        if not isinstance(gens, dict):
            gens = {}
        normalized.append(
            {
                "id": normalize_text(row.get("id")) or f"test_{i:06d}",
                "instruction": normalize_text(row.get("instruction")),
                "input": normalize_text(row.get("input")),
                "reference": normalize_text(row.get("reference")),
                "generations": {
                    "base": normalize_text(gens.get("base")),
                    "sft": normalize_text(gens.get("sft")),
                    "dpo": normalize_text(gens.get("dpo")),
                },
            }
        )
    return normalized


def evaluate_system_outputs(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_system: Dict[str, List[Dict[str, Any]]] = {k: [] for k in SYSTEMS}

    for row in rows:
        input_text = row["input"]
        rid = row["id"]
        for system in SYSTEMS:
            text = normalize_text(row.get("generations", {}).get(system))
            metrics = rule_score(input_text, text)
            per_system[system].append(
                {
                    "id": rid,
                    "coverage": metrics["coverage"],
                    "forbidden_count": metrics["forbidden_count"],
                    "repetition_ratio": metrics["repetition_ratio"],
                    "length": metrics["length"],
                    "total_score": metrics["total_score"],
                    "text": text,
                }
            )

    summary: Dict[str, Dict[str, float]] = {}
    for system, items in per_system.items():
        if not items:
            summary[system] = {
                "num_samples": 0,
                "avg_coverage": 0.0,
                "avg_forbidden_count": 0.0,
                "avg_repetition_ratio": 0.0,
                "avg_length": 0.0,
                "avg_total_score": 0.0,
            }
            continue
        summary[system] = {
            "num_samples": len(items),
            "avg_coverage": round(mean(float(x["coverage"]) for x in items), 4),
            "avg_forbidden_count": round(
                mean(float(x["forbidden_count"]) for x in items), 4
            ),
            "avg_repetition_ratio": round(
                mean(float(x["repetition_ratio"]) for x in items), 4
            ),
            "avg_length": round(mean(float(x["length"]) for x in items), 2),
            "avg_total_score": round(mean(float(x["total_score"]) for x in items), 4),
        }

    return {"summary": summary, "details": per_system}


def write_report_json(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report_csv(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "system",
        "num_samples",
        "avg_coverage",
        "avg_forbidden_count",
        "avg_repetition_ratio",
        "avg_length",
        "avg_total_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for system in SYSTEMS:
            row = {"system": system}
            row.update(summary.get(system, {}))
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 base/sft/dpo 广告文案生成质量")
    parser.add_argument(
        "--mode",
        type=str,
        default="mock",
        choices=["mock", "file"],
        help="评估模式：mock 或 file",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="测试集文件（final_test.jsonl）",
    )
    parser.add_argument(
        "--generation_file",
        type=str,
        default="",
        help="file 模式下的生成结果文件（JSON/JSONL）",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default="outputs/eval_reports/eval_report.json",
        help="评估 JSON 输出路径",
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default="outputs/eval_reports/eval_report.csv",
        help="评估 CSV 输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_rows = load_json_or_jsonl(Path(args.input_file))
    if not input_rows:
        print("[错误] 输入测试集为空，无法评估。")
        return

    if args.mode == "mock":
        eval_rows = build_mock_generations(input_rows)
        print(f"[信息] mock 模式：已构造 {len(eval_rows)} 条 base/sft/dpo 模拟输出。")
    else:
        if not args.generation_file:
            raise ValueError("file 模式需要提供 --generation_file")
        eval_rows = load_file_mode_generations(Path(args.generation_file))
        print(f"[信息] file 模式：读取生成结果 {len(eval_rows)} 条。")

    report = evaluate_system_outputs(eval_rows)
    report["meta"] = {
        "mode": args.mode,
        "input_file": args.input_file,
        "num_samples": len(eval_rows),
        "systems": SYSTEMS,
    }

    json_path = Path(args.json_output)
    csv_path = Path(args.csv_output)
    write_report_json(json_path, report)
    write_report_csv(csv_path, report["summary"])

    print(f"[完成] JSON 报告: {json_path}")
    print(f"[完成] CSV 报告: {csv_path}")
    for system in SYSTEMS:
        s = report["summary"].get(system, {})
        print(
            f"[{system}] score={s.get('avg_total_score', 0.0)} "
            f"coverage={s.get('avg_coverage', 0.0)} "
            f"forbidden={s.get('avg_forbidden_count', 0.0)}"
        )


if __name__ == "__main__":
    main()

