# Dataset Report

## 文件样本数

- `data/processed/sft_train.jsonl`: 11000
- `data/processed/sft_val.jsonl`: 1000
- `data/processed/dpo_prompt_pool.jsonl`: 6000
- `data/processed/dpo_train.jsonl`: 10758
- `data/processed/dpo_val.jsonl`: 2689
- `data/processed/dev_eval.jsonl`: 500
- `data/processed/final_test.jsonl`: 269
- `outputs/dpo/dpo_pairs_with_meta.jsonl`: 13447

## SFT 统计

- output 平均长度: 108.3
- output 最小长度: 38
- output 最大长度: 200
- instruction 字段缺失数量: 0
- input 字段缺失数量: 0
- output 字段缺失数量: 0

## DPO 统计

- chosen 平均长度: 107.28
- rejected 平均长度: 100.03
- chosen_score 平均值: 3.1878
- rejected_score 平均值: 2.1605
- chosen_score <= rejected_score 数量: 0

### chosen_source 分布

- gold: 13447

### rejected_source 分布

- exaggerated: 4744
- generic: 4543
- weak: 4160
