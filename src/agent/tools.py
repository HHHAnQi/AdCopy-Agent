import re
from typing import Any, Dict, List, Tuple

from agent.schemas import AgentState, Candidate

# 复用你已有的 rule_eval.py
from rule_eval import rule_score


FORBIDDEN_WORDS = [
    "全网第一",
    "100%",
    "百分百",
    "必买",
    "神级",
    "永久",
    "绝对",
    "最强",
    "第一",
]


def parse_attributes_from_input(input_text: str) -> Dict[str, str]:
    """
    输入示例：
    商品属性：类型=裙；颜色=黑色；材质=雪纺；风格=优雅。
    """
    text = input_text.strip()
    text = text.replace("商品属性：", "").replace("商品属性:", "")
    text = text.strip("。；; ")

    attrs = {}
    parts = re.split(r"[；;]", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k.strip()] = v.strip().strip("。")
    return attrs


def format_attributes(attrs: Dict[str, str]) -> str:
    return "；".join([f"{k}={v}" for k, v in attrs.items()])


def rule_evaluate_candidates(state: AgentState) -> Dict[str, Any]:
    """
    给所有候选打 rule_score。
    """
    input_text = state.user_input

    evaluated = []
    for cand in state.candidates:
        metrics = rule_score(input_text, cand.text)
        cand.rule_metrics = metrics
        evaluated.append(
            {
                "candidate_id": cand.candidate_id,
                "text": cand.text,
                "rule_metrics": metrics,
            }
        )

    return {
        "message": f"evaluated {len(evaluated)} candidates by rule_score",
        "candidates": evaluated,
    }


def diagnose_candidate(state: AgentState, candidate: Candidate) -> List[str]:
    """
    根据属性覆盖、违规词、长度、重复率诊断问题。
    这是 v0 规则诊断，后面可以替换成 LLM diagnose。
    """
    issues = []
    text = candidate.text
    attrs = state.attributes

    # 属性遗漏：带少量同义/层级关系的保守匹配
    missing = []
    for k, v in attrs.items():
        if not v:
            continue

        # 类型=上衣 时，如果文案或输入的衣样式已经出现衬衫/外套/卫衣等具体上衣类别，
        # 就认为类型已被隐式覆盖，不强制要求出现“上衣”二字。
        if k == "类型" and v == "上衣":
            upper_style = attrs.get("衣样式", "")
            upper_aliases = ["衬衫", "外套", "卫衣", "T恤", "针织衫", "毛衣", "开衫", "吊带", "背心"]
            if any(alias in text for alias in upper_aliases) or (upper_style and upper_style in text):
                continue

        # 类型=裙 时，如果出现“裙子/连衣裙/半身裙”等，也认为覆盖
        if k == "类型" and v == "裙":
            if any(alias in text for alias in ["裙", "裙子", "连衣裙", "半身裙"]):
                continue

        # 类型=裤 时，如果出现“裤子/短裤/阔腿裤”等，也认为覆盖
        if k == "类型" and v == "裤":
            if any(alias in text for alias in ["裤", "裤子", "短裤", "长裤", "阔腿裤", "牛仔裤"]):
                continue

        if v not in text:
            missing.append(f"{k}={v}")

    if missing:
        issues.append("missing_attributes: " + ", ".join(missing))

    for w in FORBIDDEN_WORDS:
        if w in text:
            issues.append(f"forbidden_word: {w}")

    # Detect unsupported color fabrication.
    # Example: input only says 颜色=黑色, but copy says 多种颜色 / 色彩丰富 / 黑白搭配.
    input_color = attrs.get("颜色", "")
    if input_color:
        unsupported_color_phrases = [
            "多种颜色",
            "多色",
            "色彩丰富",
            "色彩斑斓",
            "撞色",
            "拼色",
            "黑白",
            "白色",
            "彩色",
        ]

        # Conservative rule: if the input color is a single color and the copy mentions
        # additional color-style expressions not supported by input, flag it.
        if input_color == "黑色":
            for phrase in unsupported_color_phrases:
                if phrase in text and phrase != input_color:
                    issues.append(f"unsupported_color_description: input_color={input_color}, phrase={phrase}")
                    break

    m = candidate.rule_metrics or {}
    length = float(m.get("length", len(text)))
    repetition = float(m.get("repetition_ratio", 0.0))
    coverage = float(m.get("coverage", 0.0))

    if coverage < 0.75:
        issues.append(f"low_coverage: {coverage:.4f}")

    if length < 45:
        issues.append(f"too_short: {length:.0f}")

    if length > 160:
        issues.append(f"too_long: {length:.0f}")

    if repetition > 0.10:
        issues.append(f"high_repetition: {repetition:.4f}")

    return issues


def diagnose_all_candidates(state: AgentState) -> Dict[str, Any]:
    diagnosed = []

    for cand in state.candidates:
        cand.issues = diagnose_candidate(state, cand)
        diagnosed.append(
            {
                "candidate_id": cand.candidate_id,
                "issues": list(cand.issues),
            }
        )

    state.diagnosis_done = True

    return {
        "message": f"diagnosed {len(diagnosed)} candidates",
        "diagnosed": diagnosed,
    }


def combined_score(candidate: Candidate) -> float:
    """
    综合分：融合 rule_score 和 LLM-as-Judge。
    rule_total 的范围约为 0~3.5。
    judge_total 的范围约为 5~25，这里归一化到 0~3.5。
    """
    m = candidate.rule_metrics or {}
    j = candidate.judge_scores or {}

    rule_total = float(m.get("total_score", 0.0))
    coverage = float(m.get("coverage", 0.0))
    forbidden = float(m.get("forbidden_count", 0.0))
    repetition = float(m.get("repetition_ratio", 0.0))

    judge_total = float(j.get("total", 0.0))
    judge_norm = (judge_total / 25.0) * 3.5 if judge_total > 0 else 0.0

    if judge_total > 0:
        score = (
            0.55 * rule_total
            + 0.35 * judge_norm
            + 0.10 * coverage * 3.5
            - 0.50 * forbidden
            - 0.30 * repetition
        )
    else:
        score = (
            0.70 * rule_total
            + 0.25 * coverage * 3.5
            - 0.40 * forbidden
            - 0.50 * repetition
        )

    # Tie-breaker: if an LLM rewrite is equally good, prefer the revised version
    # so that the self-revision loop is visible in demo traces.
    if candidate.source == "llm_rewrite":
        score += 0.005

    candidate.combined_score = score
    return score


def select_best_candidate(state: AgentState) -> Dict[str, Any]:
    if not state.candidates:
        return {"message": "no candidates to select", "best": None}

    for cand in state.candidates:
        combined_score(cand)

    best = sorted(
        state.candidates,
        key=lambda c: c.combined_score,
        reverse=True,
    )[0]

    state.current_best_id = best.candidate_id

    return {
        "message": "selected best candidate",
        "best": {
            "candidate_id": best.candidate_id,
            "text": best.text,
            "combined_score": best.combined_score,
            "rule_metrics": best.rule_metrics,
            "judge_scores": best.judge_scores,
            "issues": best.issues,
        },
    }


def simple_rewrite_candidate(state: AgentState, candidate: Candidate) -> Candidate:
    """
    v0 简单改写：不调用模型，只做保守修复。
    后面会升级成 rewrite_copy_tool 调用 LLM。
    """
    text = candidate.text

    # 去掉明显禁用词
    for w in FORBIDDEN_WORDS:
        text = text.replace(w, "")

    # 补充遗漏属性，采用较保守的句式
    missing_attrs = []
    for issue in candidate.issues:
        if issue.startswith("missing_attributes:"):
            raw = issue.replace("missing_attributes:", "").strip()
            missing_attrs.extend([x.strip() for x in raw.split(",") if x.strip()])

    if missing_attrs:
        # 不把“类型=上衣/裙/裤”这类粗粒度类别硬塞进文案，避免出现
        # “融入上衣等设计元素”这类不自然表达。
        filtered = []
        for x in missing_attrs:
            if "=" not in x:
                continue
            k, v = x.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k == "类型":
                continue
            filtered.append(v)

        readable = "、".join(filtered)
        if readable:
            text = text.rstrip("。") + f"。同时突出{readable}等细节，丰富整体卖点。"

    # 控制过长：简单截断到句号附近
    if len(text) > 170:
        text = text[:160].rstrip("，,。") + "。"

    new_id = f"{candidate.candidate_id}_rewrite"
    return Candidate(
        candidate_id=new_id,
        text=text,
        source="rewrite",
    )


def rewrite_best_if_needed(state: AgentState) -> Dict[str, Any]:
    if not state.current_best_id:
        return {"message": "no best candidate selected", "rewritten": None}

    best = next((c for c in state.candidates if c.candidate_id == state.current_best_id), None)
    if best is None:
        return {"message": "best candidate not found", "rewritten": None}

    if not best.issues:
        return {
            "message": "best candidate has no issues, no rewrite needed",
            "rewritten": None,
        }

    rewritten = simple_rewrite_candidate(state, best)
    state.candidates.append(rewritten)

    return {
        "message": "rewritten best candidate because issues were found",
        "source_candidate_id": best.candidate_id,
        "issues": best.issues,
        "rewritten": {
            "candidate_id": rewritten.candidate_id,
            "text": rewritten.text,
        },
    }


def finish_with_best(state: AgentState) -> Dict[str, Any]:
    if not state.current_best_id:
        select_best_candidate(state)

    best = next((c for c in state.candidates if c.candidate_id == state.current_best_id), None)

    if best is None:
        state.final_copy = ""
        state.final_report = {"error": "no final candidate"}
        state.should_stop = True
        return state.final_report

    state.final_copy = best.text
    state.final_report = {
        "final_candidate_id": best.candidate_id,
        "final_copy": best.text,
        "combined_score": best.combined_score,
        "rule_metrics": best.rule_metrics,
        "judge_scores": best.judge_scores,
        "issues": best.issues,
        "source": best.source,
    }
    state.should_stop = True
    return state.final_report
