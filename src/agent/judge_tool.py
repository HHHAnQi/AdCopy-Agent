import json
import re
from typing import Any, Dict, List

import torch

from agent.schemas import AgentState, Candidate


JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for Chinese e-commerce advertisement copy. "
    "You must judge whether the generated copy faithfully follows the given product attributes. "
    "Return JSON only."
)


def _clip_score(x: Any, default: int = 3) -> int:
    try:
        v = int(float(x))
    except Exception:
        v = default
    return max(1, min(5, v))


def _extract_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()

    # 直接 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 从文本中截取 JSON
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {}


def normalize_judge_scores(raw: Dict[str, Any]) -> Dict[str, Any]:
    scores = {
        "attribute_coverage": _clip_score(raw.get("attribute_coverage", 3)),
        "factual_consistency": _clip_score(raw.get("factual_consistency", 3)),
        "naturalness": _clip_score(raw.get("naturalness", 3)),
        "attractiveness": _clip_score(raw.get("attractiveness", 3)),
        "compliance": _clip_score(raw.get("compliance", 3)),
    }
    scores["total"] = sum(scores.values())

    issues = raw.get("issues", [])
    if isinstance(issues, str):
        issues = [issues]
    if not isinstance(issues, list):
        issues = []

    scores["issues"] = [str(x) for x in issues if str(x).strip()]
    scores["comment"] = str(raw.get("comment", "")).strip()
    return scores


def build_judge_prompt(input_text: str, candidate_text: str) -> str:
    return f"""Please evaluate the following Chinese e-commerce ad copy.

Product attributes:
{input_text}

Generated copy:
{candidate_text}

Important judging rule:
Only evaluate attributes explicitly provided in the product attributes. Do not penalize the copy for not mentioning attributes that are not provided, such as length, sleeve type, collar type, brand, price, season, or target user, unless they appear in the input attributes.

Scoring dimensions, each from 1 to 5:
1. attribute_coverage: whether the copy covers the provided product attributes.
2. factual_consistency: whether the copy avoids fabricating unsupported attributes.
3. naturalness: whether the expression is fluent and natural.
4. attractiveness: whether the copy is appealing for e-commerce.
5. compliance: whether the copy avoids exaggerated or forbidden claims.

Return JSON only in this format:
{{
  "attribute_coverage": 1-5,
  "factual_consistency": 1-5,
  "naturalness": 1-5,
  "attractiveness": 1-5,
  "compliance": 1-5,
  "issues": ["short issue 1", "short issue 2"],
  "comment": "short explanation"
}}
"""


def judge_one_candidate_with_model(
    model_generator: Any,
    input_text: str,
    candidate: Candidate,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    tokenizer = model_generator.tokenizer
    model = model_generator.model

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_judge_prompt(input_text, candidate.text)},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=1536,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    raw = _extract_json(text)
    scores = normalize_judge_scores(raw)
    scores["raw_response"] = text
    return scores


def rule_based_judge_fallback(candidate: Candidate) -> Dict[str, Any]:
    """
    Fallback for mock mode or judge failure.
    This is not the main judge; it keeps the agent robust.
    """
    m = candidate.rule_metrics or {}
    coverage = float(m.get("coverage", 0.0))
    forbidden = float(m.get("forbidden_count", 0.0))
    repetition = float(m.get("repetition_ratio", 0.0))
    length = float(m.get("length", len(candidate.text)))

    attribute_coverage = 5 if coverage >= 0.95 else 4 if coverage >= 0.75 else 3 if coverage >= 0.5 else 2
    compliance = 1 if forbidden > 0 else 5
    naturalness = 4
    attractiveness = 4 if 55 <= length <= 130 else 3
    factual_consistency = 5 if coverage >= 0.75 and forbidden == 0 else 3

    if repetition > 0.1:
        naturalness = max(1, naturalness - 1)

    scores = {
        "attribute_coverage": attribute_coverage,
        "factual_consistency": factual_consistency,
        "naturalness": naturalness,
        "attractiveness": attractiveness,
        "compliance": compliance,
    }
    scores["total"] = sum(scores.values())
    scores["issues"] = []
    scores["comment"] = "fallback judge based on rule metrics"
    return scores


def judge_candidates(
    state: AgentState,
    model_generator: Any = None,
    max_candidates: int = 20,
) -> Dict[str, Any]:
    judged = []

    # Judge all candidates that have not been judged yet.
    # This is important after rewrite_copy creates a new candidate.
    candidates = [c for c in state.candidates if not c.judge_scores][:max_candidates]

    for cand in candidates:
        try:
            if model_generator is not None:
                scores = judge_one_candidate_with_model(
                    model_generator=model_generator,
                    input_text=state.user_input,
                    candidate=cand,
                )
            else:
                scores = rule_based_judge_fallback(cand)
        except Exception as e:
            scores = rule_based_judge_fallback(cand)
            scores["judge_error"] = str(e)

        cand.judge_scores = scores

        # 把 judge issues 合并到 candidate issues
        for issue in scores.get("issues", []):
            issue_text = f"judge_issue: {issue}"
            if issue_text not in cand.issues:
                cand.issues.append(issue_text)

        judged.append(
            {
                "candidate_id": cand.candidate_id,
                "judge_scores": scores,
            }
        )

    state.judge_done = True

    return {
        "message": f"judged {len(judged)} candidates",
        "judged": judged,
    }
