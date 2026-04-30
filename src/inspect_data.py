import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def load_records(file_path: Path) -> List[Dict[str, Any]]:
    """
    兼容两种格式：
    1) JSONL: 每行一个 JSON 对象
    2) JSON list: 整个文件是一个 JSON 数组
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # 先尝试按完整 JSON 读取（适配 JSON list）
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            records = [x for x in obj if isinstance(x, dict)]
            return records
        if isinstance(obj, dict):
            # 极端情况下文件是单个对象，也兼容
            return [obj]
    except json.JSONDecodeError:
        pass

    # 回退到 JSONL
    records: List[Dict[str, Any]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONL 解析失败: {file_path} 第 {idx} 行: {e}") from e
        if isinstance(item, dict):
            records.append(item)
    return records


def pick_text_fields(records: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    优先使用 content/summary；若不存在则自动推断两个最常见字段。
    """
    if not records:
        return "content", "summary"

    sample_keys = set(records[0].keys())
    if "content" in sample_keys and "summary" in sample_keys:
        return "content", "summary"

    key_freq: Dict[str, int] = {}
    for item in records:
        for k in item.keys():
            key_freq[k] = key_freq.get(k, 0) + 1
    ranked = sorted(key_freq.items(), key=lambda x: (-x[1], x[0]))

    if len(ranked) >= 2:
        return ranked[0][0], ranked[1][0]
    if len(ranked) == 1:
        return ranked[0][0], ranked[0][0]
    return "content", "summary"


def to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def text_length_stats(records: List[Dict[str, Any]], field: str) -> Tuple[float, int, int]:
    lengths: List[int] = []
    for item in records:
        lengths.append(len(to_text(item.get(field, ""))))
    if not lengths:
        return 0.0, 0, 0
    return float(mean(lengths)), min(lengths), max(lengths)


def print_dataset_report(file_path: Path) -> None:
    print("=" * 80)
    print(f"文件: {file_path}")
    try:
        records = load_records(file_path)
    except Exception as e:
        print(f"[错误] 读取失败: {e}")
        print()
        return

    print(f"数据条数: {len(records)}")
    if not records:
        print("样本为空，跳过后续统计。")
        print()
        return

    content_field, summary_field = pick_text_fields(records)
    if content_field != "content" or summary_field != "summary":
        print(
            "[提示] 未检测到标准字段 content/summary，"
            f"当前推断字段为: content='{content_field}', summary='{summary_field}'"
        )

    first_keys = sorted(records[0].keys())
    print(f"样本字段 keys（首条）: {first_keys}")

    print("\n前 3 条样本:")
    for i, item in enumerate(records[:3], start=1):
        print(f"- 样本 {i}:")
        print(json.dumps(item, ensure_ascii=False, indent=2))

    content_example = to_text(records[0].get(content_field, ""))
    summary_example = to_text(records[0].get(summary_field, ""))
    print("\ncontent 示例:")
    print(content_example[:300] if len(content_example) > 300 else content_example)
    print("\nsummary 示例:")
    print(summary_example[:300] if len(summary_example) > 300 else summary_example)

    c_avg, c_min, c_max = text_length_stats(records, content_field)
    s_avg, s_min, s_max = text_length_stats(records, summary_field)
    print(
        f"\n{content_field} 长度统计: 平均={c_avg:.2f}, 最小={c_min}, 最大={c_max}"
    )
    print(
        f"{summary_field} 长度统计: 平均={s_avg:.2f}, 最小={s_min}, 最大={s_max}"
    )
    print()


def resolve_data_paths(project_root: Path) -> Tuple[Path, Path]:
    """
    默认使用 data/train.json 和 data/dev.json。
    若不存在，则尝试 data/AdvertiseGen/train.json 和 data/AdvertiseGen/dev.json。
    """
    train_path = project_root / "data" / "train.json"
    dev_path = project_root / "data" / "dev.json"
    if train_path.exists() and dev_path.exists():
        return train_path, dev_path

    alt_train = project_root / "data" / "AdvertiseGen" / "train.json"
    alt_dev = project_root / "data" / "AdvertiseGen" / "dev.json"
    if alt_train.exists() and alt_dev.exists():
        print(
            "[提示] 未找到 data/train.json 或 data/dev.json，"
            "已自动使用 data/AdvertiseGen/train.json 和 data/AdvertiseGen/dev.json"
        )
        return alt_train, alt_dev

    return train_path, dev_path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    train_path, dev_path = resolve_data_paths(project_root)

    print_dataset_report(train_path)
    print_dataset_report(dev_path)


if __name__ == "__main__":
    main()

