import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return " ".join(str(x).replace("\n", " ").replace("\r", " ").split()).strip()


def build_messages(instruction: str, input_text: str) -> List[Dict[str, str]]:
    user = normalize_text(instruction)
    input_text = normalize_text(input_text)
    if input_text:
        user = user + "\n" + input_text

    return [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": user,
        },
    ]


def load_model(base_model: str, adapter_path: str):
    print(f"[信息] 加载 tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[信息] 加载 base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[信息] 加载 SFT adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def generate_candidates(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> List[str]:
    messages = build_messages(instruction, input_text)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    outputs = []
    with torch.no_grad():
        for _ in range(num_candidates):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_ids = out[0][input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            text = normalize_text(text)
            if text:
                outputs.append(text)

    # 去重但保序
    seen = set()
    deduped = []
    for x in outputs:
        if x not in seen:
            deduped.append(x)
            seen.add(x)
    return deduped


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data/processed/dpo_prompt_pool.jsonl")
    parser.add_argument("--output_file", default="outputs/v2/dpo_v2_candidates.jsonl")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft_adapter", default="outputs/models/qwen2_5_7b_adgen_sft")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    print(f"[信息] 输入样本数: {len(rows)}")
    print(f"[信息] 每条生成候选数: {args.num_candidates}")

    model, tokenizer = load_model(args.base_model, args.sft_adapter)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            rid = normalize_text(row.get("id")) or f"dpo_v2_{idx:06d}"
            instruction = normalize_text(row.get("instruction"))
            input_text = normalize_text(row.get("input"))
            reference = normalize_text(row.get("reference") or row.get("output"))

            candidates = generate_candidates(
                model=model,
                tokenizer=tokenizer,
                instruction=instruction,
                input_text=input_text,
                num_candidates=args.num_candidates,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

            record = {
                "id": rid,
                "instruction": instruction,
                "input": input_text,
                "reference": reference,
                "candidates": candidates,
                "meta": {
                    "source": "sft_generated",
                    "num_candidates": len(candidates),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if idx % args.progress_every == 0 or idx == len(rows):
                print(f"[进度] 已生成 {idx}/{len(rows)}")

    print(f"[完成] 写入: {output_path}")


if __name__ == "__main__":
    main()
