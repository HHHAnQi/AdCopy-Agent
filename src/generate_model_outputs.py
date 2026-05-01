import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as exc:
    raise ImportError(
        "缺少 peft，请先安装：pip install peft"
    ) from exc


def load_json_or_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {file_path}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    rows: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            print(f"[警告] 跳过无效 JSONL 行: {i}")
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def normalize_text(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_messages(instruction: str, input_text: str) -> List[Dict[str, str]]:
    user_content = normalize_text(instruction)
    input_text = normalize_text(input_text)

    if input_text:
        user_content = f"{user_content}\n{input_text}"

    return [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def load_model_and_tokenizer(
    base_model: str,
    model_type: str,
    adapter_path: Optional[str],
    torch_dtype: torch.dtype,
):
    print(f"[信息] 加载 tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[信息] 加载 base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if model_type in {"sft", "dpo"}:
        if not adapter_path:
            raise ValueError(f"model_type={model_type} 时必须提供 --adapter_path")
        adapter = Path(adapter_path)
        if not adapter.exists():
            raise FileNotFoundError(f"LoRA adapter 路径不存在: {adapter_path}")

        print(f"[信息] 加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_one(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    messages = build_messages(instruction, input_text)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=1024,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if temperature > 0:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

    gen_ids = output_ids[0][input_len:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return normalize_text(pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate outputs from Base / SFT / DPO models for evaluation."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["base", "sft", "dpo"],
        help="模型类型：base / sft / dpo",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="基础模型路径或 HuggingFace 名称",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="SFT/DPO LoRA adapter 路径。base 模式不需要。",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/processed/final_test.jsonl",
        help="输入测试集 JSONL，需包含 instruction/input/reference 字段。",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出 JSONL 文件路径。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=269,
        help="最多生成多少条。<=0 表示全部。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=180,
        help="最大生成 token 数。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度。设为 0 表示 greedy decoding。",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="top-p 采样参数。",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="重复惩罚。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="模型加载 dtype。",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=20,
        help="每多少条打印一次进度。",
    )
    return parser.parse_args()


def dtype_from_arg(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    rows = load_json_or_jsonl(input_path)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    if not rows:
        raise ValueError(f"输入样本为空: {input_path}")

    print(f"[信息] model_type: {args.model_type}")
    print(f"[信息] input_file: {input_path}")
    print(f"[信息] output_file: {output_path}")
    print(f"[信息] max_samples: {len(rows)}")

    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        model_type=args.model_type,
        adapter_path=args.adapter_path or None,
        torch_dtype=dtype_from_arg(args.dtype),
    )

    outputs: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        rid = normalize_text(row.get("id")) or f"sample_{idx:06d}"
        instruction = normalize_text(row.get("instruction"))
        input_text = normalize_text(row.get("input"))
        reference = normalize_text(row.get("reference") or row.get("output"))

        try:
            prediction = generate_one(
                model=model,
                tokenizer=tokenizer,
                instruction=instruction,
                input_text=input_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"[错误] CUDA OOM at sample {idx}. 尝试清理显存后继续。")
                torch.cuda.empty_cache()
                gc.collect()
                prediction = ""
            else:
                raise

        outputs.append(
            {
                "id": rid,
                "instruction": instruction,
                "input": input_text,
                "reference": reference,
                "prediction": prediction,
                "model_type": args.model_type,
            }
        )

        if idx % args.progress_every == 0 or idx == len(rows):
            print(f"[进度] 已生成 {idx}/{len(rows)}")

    write_jsonl(output_path, outputs)
    print(f"[完成] 写入: {output_path}")
    print(f"[完成] 样本数: {len(outputs)}")

    if torch.cuda.is_available():
        print(
            f"[显存] max allocated: "
            f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB"
        )


if __name__ == "__main__":
    main()