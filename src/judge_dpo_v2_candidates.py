import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return " ".join(str(x).replace("\n", " ").replace("\r", " ").split()).strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def build_judge_prompt(input_text: str, candidate: str) -> List[Dict[str, str]]:
    system = (
        "你是一个严格的电商广告文案评审员。"
        "你需要根据商品属性评估候选广告文案。"
        "请只输出 JSON，不要输出解释。"
    )

    user = f"""
请根据以下商品属性和候选广告文案进行评分。

商品属性：
{input_text}

候选广告文案：
{candidate}

请从 5 个维度评分，每项 1 到 5 分：
1. attribute_coverage：是否覆盖主要商品属性
2. factual_consistency：是否避免编造未提供的信息
3. naturalness：中文表达是否自然
4. attractiveness：是否有电商广告吸引力
5. compliance：是否避免夸大、绝对化、虚假承诺

请严格输出如下 JSON 格式：
{{
  "attribute_coverage": 1-5,
  "factual_consistency": 1-5,
  "naturalness": 1-5,
  "attractiveness": 1-5,
  "compliance": 1-5,
  "total": 5-25
}}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}


def safe_int(v: Any, default: int = 1) -> int:
    try:
        x = int(float(v))
        return max(1, min(5, x))
    except Exception:
        return default


def normalize_judge_result(obj: Dict[str, Any]) -> Dict[str, int]:
    keys = [
        "attribute_coverage",
        "factual_consistency",
        "naturalness",
        "attractiveness",
        "compliance",
    ]
    scores = {k: safe_int(obj.get(k), 1) for k in keys}
    scores["total"] = sum(scores.values())
    return scores


def judge_one(model, tokenizer, input_text: str, candidate: str, max_new_tokens: int) -> Dict[str, int]:
    messages = build_judge_prompt(input_text, candidate)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=2048,
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

    gen = out[0][input_len:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    obj = parse_json_from_text(text)
    return normalize_judge_result(obj)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="outputs/v2/dpo_v2_scored_candidates_100.jsonl")
    parser.add_argument("--output_file", default="outputs/v2/dpo_v2_judged_candidates_100.jsonl")
    parser.add_argument("--judge_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_candidates_per_prompt", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--progress_every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_jsonl(Path(args.input_file))

    model, tokenizer = load_model(args.judge_model)

    output_rows = []

    for idx, row in enumerate(rows, start=1):
        input_text = normalize_text(row.get("input"))
        candidates = row.get("scored_candidates", [])

        # 优先 judge rule 保留的候选，再补充其他候选
        candidates = sorted(
            candidates,
            key=lambda x: (
                x.get("keep_by_rule", False),
                x.get("rule_metrics", {}).get("total_score", 0.0),
            ),
            reverse=True,
        )[: args.max_candidates_per_prompt]

        judged = []
        for c in candidates:
            text = normalize_text(c.get("text"))
            if not text:
                continue

            judge_scores = judge_one(
                model=model,
                tokenizer=tokenizer,
                input_text=input_text,
                candidate=text,
                max_new_tokens=args.max_new_tokens,
            )

            item = dict(c)
            item["judge_scores"] = judge_scores
            judged.append(item)

        out = dict(row)
        out["judged_candidates"] = judged
        output_rows.append(out)

        if idx % args.progress_every == 0 or idx == len(rows):
            print(f"[进度] 已 judge {idx}/{len(rows)}")

    write_jsonl(Path(args.output_file), output_rows)
    print(f"[完成] 输出: {args.output_file}")


if __name__ == "__main__":
    main()
