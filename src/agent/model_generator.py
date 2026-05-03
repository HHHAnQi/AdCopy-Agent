from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.schemas import Candidate


class DPOV2ModelGenerator:
    """
    Real generator tool for AdCopy-ReAct Agent.

    It loads:
    - base model: Qwen/Qwen2.5-7B-Instruct
    - LoRA adapter: outputs/models/qwen2_5_7b_adgen_dpo_v2

    and generates multiple ad-copy candidates for a given product attribute input.
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: str = "outputs/models/qwen2_5_7b_adgen_dpo_v2",
        device_map: str = "auto",
        dtype: str = "bf16",
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

        print(f"[Generator] loading tokenizer: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[Generator] loading base model: {base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        print(f"[Generator] loading adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text or "").replace("\n", " ").replace("\r", " ").split()).strip()

    def _build_messages(self, instruction: str, input_text: str) -> List[Dict[str, str]]:
        user_content = self._normalize_text(instruction)
        input_text = self._normalize_text(input_text)
        if input_text:
            user_content = user_content + "\n" + input_text

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

    def generate(
        self,
        instruction: str,
        input_text: str,
        num_candidates: int = 4,
        max_new_tokens: int = 160,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        candidate_prefix: str = "model_cand",
    ) -> List[Candidate]:
        messages = self._build_messages(instruction, input_text)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        outputs: List[str] = []

        with torch.no_grad():
            for _ in range(num_candidates):
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                gen_ids = out[0][input_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                text = self._normalize_text(text)

                if text:
                    outputs.append(text)

        # 去重但保序
        seen = set()
        deduped = []
        for text in outputs:
            if text not in seen:
                seen.add(text)
                deduped.append(text)

        candidates = []
        for idx, text in enumerate(deduped, start=1):
            candidates.append(
                Candidate(
                    candidate_id=f"{candidate_prefix}_{idx}",
                    text=text,
                    source="dpo_v2_model_generator",
                )
            )

        return candidates
