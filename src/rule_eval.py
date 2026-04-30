import math
import re
from collections import Counter
from typing import Dict, List


FORBIDDEN_WORDS = [
    "全球第一",
    "全网第一",
    "100%",
    "永久",
    "永不",
    "必买",
    "最强",
    "神级",
    "治疗",
    "治愈",
    "减肥",
    "guaranteed",
]

TYPE_KEYWORDS = ["上衣", "裙", "裤", "鞋", "包"]
COLOR_KEYWORDS = ["白色", "黑色", "红色", "粉色", "蓝色", "绿色", "黄色", "卡其色", "深色", "浅色"]
MATERIAL_KEYWORDS = ["牛仔布", "蕾丝", "针织", "棉", "雪纺", "真丝", "皮革"]

TYPE_ALIASES = {
    "上衣": ["上衣", "衬衫", "T恤", "卫衣", "外套"],
    "裙": ["裙", "半身裙", "连衣裙", "长裙", "短裙"],
    "裤": ["裤", "裤子", "长裤", "短裤", "阔腿裤", "牛仔裤"],
    "鞋": ["鞋", "鞋子", "运动鞋", "皮鞋", "高跟鞋", "凉鞋"],
    "包": ["包", "包包", "手提包", "斜挎包", "双肩包"],
}


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def _is_chinese_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def extract_attribute_values(input_text: str) -> List[str]:
    """
    从形如 "商品属性：key=value；key=value。" 的文本中提取所有 value。
    解析失败片段自动跳过，不抛错。
    """
    text = _normalize_text(input_text)
    if not text:
        return []

    # 兼容全角/半角标点
    text = text.replace(";", "；")
    text = text.replace(":", "：")

    # 去掉前缀（若存在）
    if "商品属性：" in text:
        text = text.split("商品属性：", 1)[1]

    # 去掉结尾句号
    text = text.rstrip("。.")

    values: List[str] = []
    for segment in text.split("；"):
        seg = segment.strip()
        if not seg or "=" not in seg:
            continue
        _, value = seg.split("=", 1)
        value = value.strip()
        if value:
            values.append(value)
    return values


def extract_attribute_map(input_text: str) -> Dict[str, str]:
    """
    从 "商品属性：key=value；..." 解析键值映射。
    """
    text = _normalize_text(input_text)
    if not text:
        return {}

    text = text.replace(";", "；").replace(":", "：")
    if "商品属性：" in text:
        text = text.split("商品属性：", 1)[1]
    text = text.rstrip("。.")

    attrs: Dict[str, str] = {}
    for segment in text.split("；"):
        seg = segment.strip()
        if not seg or "=" not in seg:
            continue
        key, value = seg.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            attrs[key] = value
    return attrs


def compute_coverage(input_text: str, text: str) -> float:
    """
    计算属性 value 在生成文案中的覆盖率（0~1）。
    """
    values = extract_attribute_values(input_text)
    if not values:
        return 0.0

    target = _normalize_text(text)
    if not target:
        return 0.0

    hit = 0
    for v in values:
        if v and v in target:
            hit += 1
    return hit / len(values)


def detect_forbidden_words(text: str) -> List[str]:
    """
    检测夸大/违规词，返回命中的词列表（去重，按 FORBIDDEN_WORDS 顺序）。
    """
    target = _normalize_text(text).lower()
    if not target:
        return []

    hits: List[str] = []
    for w in FORBIDDEN_WORDS:
        token = w.lower().strip()
        if not token:
            continue
        if token in target:
            hits.append(w)
    return hits


def detect_attribute_conflicts(input_text: str, text: str) -> List[str]:
    """
    检测 input 属性与生成文案的事实冲突。
    """
    attrs = extract_attribute_map(input_text)
    target = _normalize_text(text)
    if not attrs or not target:
        return []

    conflicts: List[str] = []

    # 1) 类型冲突
    input_type = attrs.get("类型", "").strip()
    if input_type in TYPE_KEYWORDS:
        allowed_tokens = TYPE_ALIASES.get(input_type, [input_type])
        input_type_hit = any(token in target for token in allowed_tokens)
        for t in TYPE_KEYWORDS:
            if t == input_type:
                continue
            other_tokens = TYPE_ALIASES.get(t, [t])
            if any(token in target for token in other_tokens):
                # 只有当文本里也显式出现了输入类型或其别名时，才严格判冲突
                # 若仅出现其他类型词，依然记为可疑冲突（更保守）
                if input_type_hit:
                    conflicts.append(f"类型冲突: 输入={input_type}, 文案出现={t}")
                else:
                    conflicts.append(f"类型疑似冲突: 输入={input_type}, 文案出现={t}")
                break

    # 2) 颜色冲突
    input_color = attrs.get("颜色", "").strip()
    if input_color:
        if input_color == "深色":
            if ("白色" in target) or ("浅色" in target):
                conflicts.append("颜色冲突: 输入=深色, 文案出现=白色/浅色")
        elif input_color == "浅色":
            if ("黑色" in target) or ("深色" in target):
                conflicts.append("颜色冲突: 输入=浅色, 文案出现=黑色/深色")
        elif input_color in COLOR_KEYWORDS:
            for c in COLOR_KEYWORDS:
                if c == input_color:
                    continue
                if c in target:
                    conflicts.append(f"颜色冲突: 输入={input_color}, 文案出现={c}")
                    break

    # 3) 材质冲突
    input_material = attrs.get("材质", "").strip()
    if input_material in MATERIAL_KEYWORDS:
        for m in MATERIAL_KEYWORDS:
            if m == input_material:
                continue
            if m in target:
                conflicts.append(f"材质冲突: 输入={input_material}, 文案出现={m}")
                break

    return conflicts


def repetition_ratio(text: str, n: int = 3) -> float:
    """
    计算 n-gram 重复率：
    重复 n-gram 数量 / n-gram 总数，范围 0~1。
    中文场景优先统计中文字符序列。
    """
    if n <= 0:
        return 0.0

    target = _normalize_text(text)
    chars = [c for c in target if _is_chinese_char(c)]

    if len(chars) < n:
        return 0.0

    grams = ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]
    total = len(grams)
    if total == 0:
        return 0.0

    counter = Counter(grams)
    duplicate_count = sum(c - 1 for c in counter.values() if c > 1)
    ratio = duplicate_count / total
    return max(0.0, min(1.0, ratio))


def length_score(text: str) -> float:
    """
    长度评分（0~1）：
    - 60~160 字符区间得分最高（1.0）
    - 过短/过长按距离平滑衰减
    """
    target = _normalize_text(text)
    length = len([c for c in target if _is_chinese_char(c)])

    if length == 0:
        return 0.0
    if 60 <= length <= 160:
        return 1.0

    # 距离越远衰减越明显，使用指数衰减保证平滑
    if length < 60:
        distance = 60 - length
    else:
        distance = length - 160

    score = math.exp(-distance / 35.0)
    return max(0.0, min(1.0, score))


def rule_score(input_text: str, text: str) -> Dict[str, object]:
    """
    综合规则评分，返回详细子项与总分（0~5）。
    total_score:
    - coverage 权重 0.45
    - length_score 权重 0.25
    - repetition 惩罚 0.15
    - forbidden 惩罚 0.15
    """
    target = _normalize_text(text)
    cn_len = len([c for c in target if _is_chinese_char(c)])

    cov = compute_coverage(input_text, target)
    forbidden = detect_forbidden_words(target)
    conflicts = detect_attribute_conflicts(input_text, target)
    rep = repetition_ratio(target, n=3)
    len_s = length_score(target)

    # 惩罚项映射到 [0,1]，越大惩罚越重
    repetition_penalty = max(0.0, min(1.0, rep))
    forbidden_penalty = min(1.0, len(forbidden) / 3.0)
    conflict_penalty = min(1.0, len(conflicts) / 2.0)

    weighted = (
        0.45 * cov
        + 0.25 * len_s
        - 0.15 * repetition_penalty
        - 0.15 * forbidden_penalty
        - 0.2 * conflict_penalty
    )

    # 映射到 0~5
    total = max(0.0, min(5.0, weighted * 5.0))

    return {
        "coverage": round(cov, 4),
        "forbidden_count": len(forbidden),
        "forbidden_words": forbidden,
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "repetition_ratio": round(rep, 4),
        "length": cn_len,
        "length_score": round(len_s, 4),
        "total_score": round(total, 4),
    }


def main() -> None:
    demo_input = "商品属性：类型=裙；版型=宽松；风格=通勤；图案=纯色。"
    demo_text = (
        "这款宽松A字裙采用纯色设计，通勤穿搭非常省心。"
        "版型利落显气质，面料舒适亲肤，日常上班与约会都能轻松驾驭。"
    )

    result = rule_score(demo_input, demo_text)
    print("=== Rule Eval Demo ===")
    print(f"input: {demo_input}")
    print(f"text: {demo_text}")
    print("result:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

