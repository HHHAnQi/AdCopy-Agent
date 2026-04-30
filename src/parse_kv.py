import re
from typing import Dict


def _clean_text(text: str) -> str:
    """
    清理输入文本：
    - 去除首尾空白
    - 去除换行与制表符
    - 压缩多余空白
    - 过滤常见异常控制字符（保留常规可见字符）
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # 去掉不可见控制字符
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    # 压缩连续空白
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_advertisegen_content(content: str) -> Dict[str, str]:
    """
    解析 AdvertiseGen 的 content 字段为字典。

    规则：
    - 用 '*' 分割多个属性片段
    - 每个片段用 '#' 分割 key/value
    - 解析失败的片段直接跳过，不抛错

    示例：
    "类型#裙*版型#宽松*风格#通勤" -> {"类型": "裙", "版型": "宽松", "风格": "通勤"}
    """
    cleaned = _clean_text(content)
    if not cleaned:
        return {}

    attrs: Dict[str, str] = {}
    parts = cleaned.split("*")
    for part in parts:
        piece = _clean_text(part)
        if not piece or "#" not in piece:
            continue

        # 只在第一个 # 处分割，避免 value 内再次出现 # 导致错误拆分
        key, value = piece.split("#", 1)
        key = _clean_text(key)
        value = _clean_text(value)
        if not key or not value:
            continue

        attrs[key] = value

    return attrs


def format_attributes(attrs: Dict[str, str]) -> str:
    """
    将属性字典格式化为自然语言描述。

    输出示例：
    商品属性：类型=裙；版型=宽松；风格=通勤；图案=纯色。
    """
    if not attrs:
        return "商品属性：无。"

    kv_text = "；".join(f"{k}={v}" for k, v in attrs.items())
    return f"商品属性：{kv_text}。"


def _demo() -> None:
    examples = [
        "类型#裙*版型#宽松*风格#通勤*图案#纯色*裙型#A字裙",
        " 类型#上衣 * 风格#简约 \n* 图案#刺绣 ",
        "类型#裤*异常片段*颜色#黑色*无效#*#缺失key",
        "",
    ]

    print("=== parse_advertisegen_content 示例 ===")
    for i, content in enumerate(examples, start=1):
        parsed = parse_advertisegen_content(content)
        formatted = format_attributes(parsed)
        print(f"\n示例 {i}")
        print(f"原始 content: {repr(content)}")
        print(f"解析结果: {parsed}")
        print(f"格式化结果: {formatted}")


if __name__ == "__main__":
    _demo()

