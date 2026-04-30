import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


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


def extract_attr_values(input_text: str) -> List[str]:
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


def weaken_reference(reference: str, input_text: str) -> str:
    """
    弱化版本：
    - 从 reference 中移除部分属性值词
    - 若移除后太短，补一条中性句
    """
    text = normalize_text(reference)
    values = extract_attr_values(input_text)
    if not text:
        return "这款商品整体风格简洁，穿着体验舒适，适合多种日常场景。"

    removable = values[::2]  # 每隔一个去掉一个
    for v in removable:
        if len(v) >= 2:
            text = text.replace(v, "")

    text = re.sub(r"[，。；]{2,}", "。", text)
    text = text.strip("，。； ")
    if len(text) < 20:
        text = "这款商品整体设计简洁，材质舒适，日常穿搭省心。"
    if not text.endswith("。"):
        text += "。"
    return text


def exaggerated_version(reference: str) -> str:
    base = normalize_text(reference)
    if not base:
        base = "这款商品设计出众，穿搭体验舒适。"
    extra = "全网第一，效果100%，必买神级单品，品质永久 guaranteed。"
    if base.endswith("。"):
        return f"{base}{extra}"
    return f"{base}。{extra}"


def generic_version(input_text: str) -> str:
    values = extract_attr_values(input_text)
    if values:
        hint = "、".join(values[:2])
        return (
            f"这是一款适合日常使用的时尚单品，整体风格自然百搭，"
            f"围绕{hint}等特点进行搭配更出彩，轻松满足多场景穿搭需求。"
        )
    return "这是一款适合日常使用的时尚单品，设计简洁百搭，轻松满足多场景需求。"


def build_mock_candidates(record: Dict[str, Any], num_candidates: int) -> List[Dict[str, str]]:
    reference = normalize_text(record.get("reference") or record.get("output"))
    input_text = normalize_text(record.get("input"))

    candidates = [
        {"source": "gold", "text": reference},
        {"source": "weak", "text": weaken_reference(reference, input_text)},
        {"source": "exaggerated", "text": exaggerated_version(reference)},
        {"source": "generic", "text": generic_version(input_text)},
    ]

    if num_candidates <= len(candidates):
        return candidates[:num_candidates]

    # 若请求更多候选，补充 generic 变体
    for i in range(len(candidates) + 1, num_candidates + 1):
        candidates.append(
            {
                "source": f"generic_{i}",
                "text": generic_version(input_text),
            }
        )
    return candidates


def load_transformers_model(model_name_or_path: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "transformers 模式依赖未安装，请先安装 transformers/torch。"
        ) from e

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
    except Exception as e:
        raise RuntimeError(
            "模型加载失败，请检查 model_name_or_path、网络环境与本地显存/内存。"
            f"\nmodel_name_or_path={model_name_or_path}\n原始错误: {e}"
        ) from e
    return tokenizer, model


def generate_with_transformers(
    record: Dict[str, Any],
    tokenizer,
    model,
    num_candidates: int,
) -> List[Dict[str, str]]:
    import torch

    instruction = normalize_text(record.get("instruction"))
    input_text = normalize_text(record.get("input"))
    prompt = f"{instruction}\n{input_text}\n请输出广告文案："

    candidates: List[Dict[str, str]] = []
    for idx in range(num_candidates):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=True,
                temperature=0.8 + random.random() * 0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if not text:
            text = "这款商品风格鲜明，细节设计到位，能够满足日常穿搭与多场景使用需求。"
        candidates.append({"source": f"model_{idx+1}", "text": text})
    return candidates


def convert_record(record: Dict[str, Any], idx: int) -> Dict[str, Any]:
    rid = normalize_text(record.get("id"))
    if not rid:
        rid = f"sample_{idx:06d}"
    return {
        "id": rid,
        "instruction": normalize_text(record.get("instruction")),
        "input": normalize_text(record.get("input")),
        "reference": normalize_text(record.get("reference") or record.get("output")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 DPO 构造候选广告文案")
    parser.add_argument("--input_file", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument(
        "--mode",
        type=str,
        default="mock",
        choices=["mock", "transformers"],
        help="候选构造模式",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL,
        help="transformers 模式使用的模型名或路径",
    )
    parser.add_argument("--max_samples", type=int, default=1000, help="最多处理样本数")
    parser.add_argument("--num_candidates", type=int, default=4, help="每条样本候选数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    rows = load_json_or_jsonl(input_path)
    rows = rows[: max(0, args.max_samples)]
    base_rows = [convert_record(r, i + 1) for i, r in enumerate(rows)]

    tokenizer = None
    model = None
    if args.mode == "transformers":
        print(f"[信息] 正在加载模型: {args.model_name_or_path}")
        tokenizer, model = load_transformers_model(args.model_name_or_path)
        print("[信息] 模型加载成功，开始生成候选。")

    output_rows: List[Dict[str, Any]] = []
    for i, r in enumerate(base_rows, start=1):
        if args.mode == "mock":
            candidates = build_mock_candidates(r, num_candidates=args.num_candidates)
        else:
            candidates = generate_with_transformers(
                r,
                tokenizer=tokenizer,
                model=model,
                num_candidates=args.num_candidates,
            )

        output_rows.append(
            {
                "id": r["id"],
                "instruction": r["instruction"],
                "input": r["input"],
                "reference": r["reference"],
                "candidates": candidates,
            }
        )

        if i % 50 == 0:
            print(f"[进度] 已处理 {i}/{len(base_rows)}")

    write_jsonl(output_path, output_rows)
    print(f"[完成] 写入文件: {output_path}")
    print(f"[完成] 样本数: {len(output_rows)}")


if __name__ == "__main__":
    main()

