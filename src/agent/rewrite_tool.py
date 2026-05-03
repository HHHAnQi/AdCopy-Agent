from typing import Any, Dict, List

import torch

from agent.schemas import AgentState, Candidate
from agent.tools import simple_rewrite_candidate


REWRITE_SYSTEM_PROMPT = (
    "You are a professional Chinese e-commerce copywriter and compliance editor. "
    "Your task is to revise an ad copy according to product attributes and detected issues. "
    "Return the revised Chinese ad copy only. Do not explain."
)


def build_rewrite_prompt(
    input_text: str,
    candidate_text: str,
    issues: List[str],
    judge_scores: Dict[str, Any],
) -> str:
    issue_text = "\n".join([f"- {x}" for x in issues]) if issues else "- No explicit issue"

    judge_comment = ""
    if judge_scores:
        judge_comment = str(judge_scores.get("comment", "")).strip()

    return f"""Please revise the following Chinese e-commerce ad copy.

Product attributes:
{input_text}

Original copy:
{candidate_text}

Detected issues:
{issue_text}

Judge comment:
{judge_comment}

Revision requirements:
1. Keep the copy faithful to the provided product attributes.
2. Do not add unsupported attributes such as unprovided color, length, brand, season, price, target user, or fabric.
3. Cover the important provided attributes as much as possible.
4. Remove exaggerated or forbidden claims.
5. Keep the style natural, attractive, and concise.
6. Output only the revised Chinese ad copy. Do not include explanations.
"""


def clean_rewrite_output(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split()).strip()

    # 常见模型前缀清理
    prefixes = [
        "改写后的文案：",
        "改写文案：",
        "修改后的文案：",
        "优化后的文案：",
        "文案：",
        "输出：",
    ]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    # 去掉引号
    text = text.strip("“”\"' ")
    return text


def rewrite_candidate_with_model(
    model_generator: Any,
    state: AgentState,
    candidate: Candidate,
    max_new_tokens: int = 180,
) -> Candidate:
    tokenizer = model_generator.tokenizer
    model = model_generator.model

    prompt_text = build_rewrite_prompt(
        input_text=state.user_input,
        candidate_text=candidate.text,
        issues=candidate.issues,
        judge_scores=candidate.judge_scores,
    )

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
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
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = clean_rewrite_output(text)

    if not text:
        raise ValueError("empty rewrite output")

    return Candidate(
        candidate_id=f"{candidate.candidate_id}_rewrite",
        text=text,
        source="llm_rewrite",
    )


def rewrite_best_candidate(
    state: AgentState,
    model_generator: Any = None,
) -> Dict[str, Any]:
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

    try:
        if model_generator is not None:
            rewritten = rewrite_candidate_with_model(
                model_generator=model_generator,
                state=state,
                candidate=best,
            )
        else:
            rewritten = simple_rewrite_candidate(state, best)
    except Exception as e:
        rewritten = simple_rewrite_candidate(state, best)
        rewritten.source = "simple_rewrite_fallback"
        error = str(e)
    else:
        error = None

    state.candidates.append(rewritten)

    return {
        "message": "rewritten best candidate because issues were found",
        "rewrite_mode": rewritten.source,
        "source_candidate_id": best.candidate_id,
        "issues": list(best.issues),
        "rewrite_error": error,
        "rewritten": {
            "candidate_id": rewritten.candidate_id,
            "text": rewritten.text,
            "source": rewritten.source,
        },
    }
