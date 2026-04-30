import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from parse_kv import format_attributes, parse_advertisegen_content
from rule_eval import repetition_ratio


INSTRUCTION = (
    "请根据商品属性生成一段中文电商广告文案，要求突出核心卖点，"
    "表达自然，有购买吸引力，不得编造未提供的信息。"
)


def load_records(file_path: Path) -> List[Dict[str, Any]]:
    """
    兼容 JSON list 与 JSONL 两种输入格式。
    """
    if not file_path.exists():
        return []

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # 先尝试按完整 JSON 读取（JSON list）
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # 回退 JSONL
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


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def clean_summary_text(value: Any) -> str:
    """
    清理 summary 文本：
    - 将 wan全 替换为 完全
    - 将 BRAND 替换为空
    - 规范空白字符
    """
    text = normalize_text(value)
    text = text.replace("wan全", "完全")
    text = text.replace("BRAND", "")
    text = " ".join(text.split())
    return text.strip()


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    zh = [c for c in chars if "\u4e00" <= c <= "\u9fff"]
    return len(zh) / len(chars)


def has_too_many_repeated_words(text: str) -> bool:
    """
    简单重复词检测（无分词依赖）：
    - 提取连续中英文词片段，统计重复占比；
    - 或出现明显短语连续重复（如 xx xx xx）。
    """
    if not text:
        return False

    if re.search(r"(.{2,6})\1{2,}", text):
        return True

    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text)
    if len(tokens) < 6:
        return False
    counter = Counter(tokens)
    repeated = sum(v for v in counter.values() if v >= 3)
    return (repeated / len(tokens)) > 0.35


def filter_and_transform(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    过滤异常样本并转换为统一中间格式：
    {
      "instruction": "...",
      "input": "商品属性：...",
      "output": "...",
      "raw_content": "..."
    }
    """
    transformed: List[Dict[str, str]] = []
    reason_counter: Dict[str, int] = {
        "empty_content_or_summary": 0,
        "contains_UNK": 0,
        "summary_len_out_of_range": 0,
        "content_attr_too_few": 0,
        "low_chinese_ratio": 0,
        "too_many_repeated_words": 0,
        "high_repetition_ratio": 0,
    }

    for item in records:
        content = normalize_text(item.get("content", ""))
        raw_summary = normalize_text(item.get("summary", ""))
        summary = clean_summary_text(raw_summary)

        if not content or not summary:
            reason_counter["empty_content_or_summary"] += 1
            continue

        if "<UNK>" in raw_summary:
            reason_counter["contains_UNK"] += 1
            continue

        if len(summary) < 30 or len(summary) > 220:
            reason_counter["summary_len_out_of_range"] += 1
            continue

        attrs = parse_advertisegen_content(content)
        if len(attrs) < 2:
            reason_counter["content_attr_too_few"] += 1
            continue

        if chinese_ratio(summary) < 0.6:
            reason_counter["low_chinese_ratio"] += 1
            continue

        if has_too_many_repeated_words(summary):
            reason_counter["too_many_repeated_words"] += 1
            continue

        if repetition_ratio(summary, n=3) > 0.25:
            reason_counter["high_repetition_ratio"] += 1
            continue

        sample = {
            "instruction": INSTRUCTION,
            "input": format_attributes(attrs),
            "output": summary,
            "raw_content": content,
        }
        transformed.append(sample)
    return transformed, reason_counter


def deduplicate(samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    先对 summary 去重，再对 (content + summary) 去重。
    """
    summary_seen = set()
    stage1: List[Dict[str, str]] = []
    for s in samples:
        key = s["output"]
        if key in summary_seen:
            continue
        summary_seen.add(key)
        stage1.append(s)

    pair_seen = set()
    final: List[Dict[str, str]] = []
    for s in stage1:
        key = (s["raw_content"], s["output"])
        if key in pair_seen:
            continue
        pair_seen.add(key)
        final.append(s)
    return final


def write_jsonl(file_path: Path, rows: List[Dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sft_rows(samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "instruction": s["instruction"],
            "input": s["input"],
            "output": s["output"],
        }
        for s in samples
    ]


def build_eval_rows(samples: List[Dict[str, str]], prefix: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for i, s in enumerate(samples, start=1):
        rows.append(
            {
                "id": f"{prefix}_{i:06d}",
                "instruction": s["instruction"],
                "input": s["input"],
                "reference": s["output"],
                "raw_content": s["raw_content"],
            }
        )
    return rows


def resolve_input_paths(project_root: Path) -> Tuple[Path, Path]:
    alt_train = project_root / "data" / "AdvertiseGen" / "train.json"
    alt_dev = project_root / "data" / "AdvertiseGen" / "dev.json"
    if alt_train.exists() and alt_dev.exists():
        return alt_train, alt_dev

    train = project_root / "data" / "train.json"
    dev = project_root / "data" / "dev.json"
    if train.exists() and dev.exists():
        print(
            "[提示] 未找到 data/AdvertiseGen/train.json 或 data/AdvertiseGen/dev.json，"
            "已回退使用 data/train.json 与 data/dev.json"
        )
        return train, dev

    return alt_train, alt_dev


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path, dev_path = resolve_input_paths(project_root)
    train_raw = load_records(train_path)
    dev_raw = load_records(dev_path)

    train_filtered, train_reason_counter = filter_and_transform(train_raw)
    dev_filtered, dev_reason_counter = filter_and_transform(dev_raw)

    train_samples = deduplicate(train_filtered)
    dev_samples = deduplicate(dev_filtered)

    # train 正式划分（不足则按实际数量），各子集天然不重合
    sft_train_samples = train_samples[:11000]
    sft_val_samples = train_samples[11000:12000]
    dpo_prompt_pool_samples = train_samples[12000:18000]

    # dev 仅用于评估，不参与训练
    dev_eval_samples = dev_samples[:500]
    final_test_samples = dev_samples[500:1000]

    sft_train_rows = build_sft_rows(sft_train_samples)
    sft_val_rows = build_sft_rows(sft_val_samples)
    dpo_prompt_pool_rows = build_eval_rows(dpo_prompt_pool_samples, prefix="dpo")
    dev_eval_rows = build_eval_rows(dev_eval_samples, prefix="dev")
    final_test_rows = build_eval_rows(final_test_samples, prefix="test")

    sft_train_path = output_dir / "sft_train.jsonl"
    sft_val_path = output_dir / "sft_val.jsonl"
    dpo_prompt_pool_path = output_dir / "dpo_prompt_pool.jsonl"
    dev_eval_path = output_dir / "dev_eval.jsonl"
    final_test_path = output_dir / "final_test.jsonl"

    write_jsonl(sft_train_path, sft_train_rows)
    write_jsonl(sft_val_path, sft_val_rows)
    write_jsonl(dpo_prompt_pool_path, dpo_prompt_pool_rows)
    write_jsonl(dev_eval_path, dev_eval_rows)
    write_jsonl(final_test_path, final_test_rows)

    print(f"生成完成: {sft_train_path} -> {len(sft_train_rows)}")
    print(f"生成完成: {sft_val_path} -> {len(sft_val_rows)}")
    print(f"生成完成: {dpo_prompt_pool_path} -> {len(dpo_prompt_pool_rows)}")
    print(f"生成完成: {dev_eval_path} -> {len(dev_eval_rows)}")
    print(f"生成完成: {final_test_path} -> {len(final_test_rows)}")

    print("\n[过滤统计] train")
    print(f"- 原始数量: {len(train_raw)}")
    print(f"- 过滤后数量(去重前): {len(train_filtered)}")
    print(f"- 去重后数量: {len(train_samples)}")
    for reason, cnt in sorted(train_reason_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {reason}: {cnt}")

    print("\n[过滤统计] dev")
    print(f"- 原始数量: {len(dev_raw)}")
    print(f"- 过滤后数量(去重前): {len(dev_filtered)}")
    print(f"- 去重后数量: {len(dev_samples)}")
    for reason, cnt in sorted(dev_reason_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {reason}: {cnt}")


if __name__ == "__main__":
    main()

