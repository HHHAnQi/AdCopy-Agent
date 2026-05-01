import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "outputs/models/qwen2_5_7b_adgen_sft"
TEST_FILE = "data/processed/final_test.jsonl"


def load_samples(path, n=5):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def build_prompt(instruction, input_text):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": instruction + "\n" + input_text},
    ]
    return messages


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    samples = load_samples(TEST_FILE, n=5)

    for i, row in enumerate(samples, 1):
        messages = build_prompt(row["instruction"], row["input"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        print("=" * 80)
        print(f"[样本 {i}]")
        print("INPUT:")
        print(row["input"])
        print("\nREFERENCE:")
        print(row.get("reference", ""))
        print("\nSFT OUTPUT:")
        print(pred)


if __name__ == "__main__":
    main()