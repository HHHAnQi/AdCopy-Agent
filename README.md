# LLM Post-Training System for E-commerce Ad Copy Generation

A Qwen2.5-based post-training system for Chinese e-commerce advertisement copy generation.

This project builds a complete LLM application and post-training pipeline, including:

- AdvertiseGen data preprocessing
- SFT dataset construction
- Rule-based DPO v1 preference pair construction
- Model-generated DPO v2 preference pair construction
- LLM-as-Judge preference filtering
- Qwen2.5-7B QLoRA SFT
- DPO preference alignment
- Base / SFT / DPO v1 / DPO v2 generation evaluation
- FastAPI demo service

## 1. Project Motivation

E-commerce ad copy generation requires more than fluent text generation. The model should:

- Cover key product attributes
- Avoid unsupported claims
- Avoid exaggerated or forbidden expressions
- Produce concise and attractive copy
- Follow business-specific style constraints

This project explores how SFT and DPO can improve a base LLM for product attribute-to-ad-copy generation.

## 2. Pipeline

```text
AdvertiseGen raw data
    ↓
Attribute parsing and text cleaning
    ↓
SFT dataset construction
    ↓
Rule-based DPO v1 pair construction
    ↓
Qwen2.5-7B QLoRA SFT
    ↓
DPO v1 alignment
    ↓
SFT-generated candidates for DPO v2
    ↓
rule_score + LLM-as-Judge filtering
    ↓
DPO v2-small alignment
    ↓
Base / SFT / DPO v1 / DPO v2 evaluation
    ↓
FastAPI demo
```

## 3. Data

Processed v1 data statistics:

| Split | Size |
|-------|-----:|
| SFT train | 11,000 |
| SFT validation | 1,000 |
| DPO prompt pool | 6,000 |
| DPO v1 train | 10,758 |
| DPO v1 validation | 2,689 |
| Dev eval | 500 |
| Final test | 269 |

DPO v1 contains three types of rejected responses:

- **exaggerated**: exaggerated or forbidden expressions
- **generic**: generic template-like copy
- **weak**: weakened copy with missing selling points

## 4. DPO v2-small

DPO v2-small uses model-generated candidates and LLM-as-Judge filtering.

Process:

```text
SFT model generates 5 candidates per prompt
→ rule_score evaluates candidate quality
→ LLM-as-Judge scores each candidate
→ high-confidence chosen/rejected pairs are selected
```

## DPO v2-small Data Statistics

| Item | Count |
|------|------:|
| Input prompts | 1,000 |
| Prompts with high-confidence pairs | 379 |
| Total pairs | 379 |
| Train pairs | 323 |
| Validation pairs | 56 |
| Min score gap | 0.4 |
| Max pairs per prompt | 1 |

## Pair Quality on the Training Split

| Metric | Value |
|--------|------:|
| Avg score gap | 0.7196 |
| Avg chosen rule score | 3.2285 |
| Avg rejected rule score | 2.6699 |
| Avg chosen judge score | 23.7492 |
| Avg rejected judge score | 19.2260 |
| Judge chosen > rejected ratio | 1.0000 |
| Rule chosen > rejected ratio | 1.0000 |

## 5. Training

Base model:

Qwen/Qwen2.5-7B-Instruct

Training method:

QLoRA / LoRA

### SFT

| Metric | Value |
|--------|------:|
| Epochs | 2 |
| Steps | 2,750 |
| Train loss | 2.5848 |
| Runtime | 7,045.76s |

### DPO v1

| Metric | Value |
|--------|------:|
| Epochs | 1 |
| Steps | 625 |
| Train loss | 0.0156 |
| Runtime | 3,247.98s |

### DPO v2-small

| Metric | Value |
|--------|------:|
| Epochs | 1 |
| Steps | 41 |
| Train loss | 1.8015 |
| Runtime | 230.31s |

## 6. Evaluation

Final-test size: 269 samples.

| Model | Coverage | Forbidden Count | Repetition Ratio | Avg Length | Rule Score |
|-------|---------:|----------------:|-----------------:|-----------:|------------:|
| Base Qwen2.5-7B-Instruct | 0.9260 | 0.0186 | 0.0238 | 191.9963 | 2.4888 |
| SFT | 0.8568 | 0.0000 | 0.0140 | 81.9814 | 3.0921 |
| DPO v1 | 0.8766 | 0.0000 | 0.0215 | 60.2119 | 3.0177 |
| DPO v2-small | 0.8652 | 0.0000 | 0.0156 | 79.0855 | 3.1371 |

## 7. Interpretation

- **SFT** is the main improvement over the base model. It improves the overall rule score, removes forbidden expressions, reduces repetition, and shortens the output length.
- **DPO v1** improves attribute coverage but tends to over-compress generation, resulting in shorter outputs and a slightly lower rule score than SFT.
- **DPO v2-small** uses model-generated candidates and LLM-as-Judge high-confidence filtering. It achieves the best overall rule score, 3.1371, while keeping forbidden expressions at zero and maintaining a natural average length close to SFT.

## 8. Repository Structure

```text
app/                     FastAPI demo
configs/                 LLaMA-Factory configs
scripts/                 Training scripts
src/                     Data processing, generation, judging, and evaluation scripts
docs/                    Experiment summary and comparison samples
docs/results/            Evaluation CSV and dataset report
```

## 9. Reproduce Data Processing

```bash
python src/inspect_data.py
python src/prepare_sft.py

python src/generate_candidates.py \
  --input_file data/processed/dpo_prompt_pool.jsonl \
  --output_file outputs/candidates/candidates.jsonl \
  --mode mock \
  --max_samples 6000 \
  --num_candidates 7

python src/build_dpo_pairs.py \
  --input_file outputs/candidates/candidates.jsonl \
  --train_output data/processed/dpo_train.jsonl \
  --val_output data/processed/dpo_val.jsonl \
  --scored_output outputs/dpo/dpo_scored_candidates.jsonl

python src/dataset_report.py
```

## 10. Reproduce DPO v2-small Data

```bash
python src/generate_dpo_v2_candidates.py \
  --input_file data/processed/dpo_prompt_pool.jsonl \
  --output_file outputs/v2/dpo_v2_candidates_1000.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --sft_adapter outputs/models/qwen2_5_7b_adgen_sft \
  --max_samples 1000 \
  --num_candidates 5 \
  --temperature 0.9 \
  --top_p 0.9

python src/score_dpo_v2_candidates.py \
  --input_file outputs/v2/dpo_v2_candidates_1000.jsonl \
  --output_file outputs/v2/dpo_v2_scored_candidates_1000.jsonl

python src/judge_dpo_v2_candidates.py \
  --input_file outputs/v2/dpo_v2_scored_candidates_1000.jsonl \
  --output_file outputs/v2/dpo_v2_judged_candidates_1000.jsonl \
  --judge_model Qwen/Qwen2.5-7B-Instruct \
  --max_candidates_per_prompt 5

python src/build_dpo_v2_pairs.py \
  --input_file outputs/v2/dpo_v2_judged_candidates_1000.jsonl \
  --train_output data/processed/dpo_v2_train_1000_gap04.jsonl \
  --val_output data/processed/dpo_v2_val_1000_gap04.jsonl \
  --stats_output outputs/v2/dpo_v2_pair_stats_1000_gap04.json \
  --min_score_gap 0.4 \
  --max_pairs_per_prompt 1
```

## 11. Training

```bash
bash scripts/train_sft.sh
bash scripts/train_dpo.sh
bash scripts/train_dpo_v2.sh
```

## 12. Limitations

- DPO v2-small uses only 379 high-confidence preference pairs.
- LLM-as-Judge may contain bias and does not replace human evaluation.
- Rule-based evaluation cannot fully capture ad-copy attractiveness or real business conversion.
- Human preference evaluation is not yet included.

## 13. Future Work

- Scale DPO v2 to 2,000–3,000 prompts.
- Add human preference evaluation.
- Improve LLM-as-Judge prompts for stricter factual consistency.
- Support multiple copywriting styles and business constraints.