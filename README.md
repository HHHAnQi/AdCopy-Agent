# LLM Post-Training System for E-commerce Ad Copy Generation

A Qwen2.5-based post-training system for Chinese e-commerce advertisement copy generation.

This project builds a complete LLM application and post-training pipeline, including:

- AdvertiseGen data preprocessing
- SFT dataset construction
- Rule-based DPO preference pair construction
- Qwen2.5-7B QLoRA SFT
- DPO preference alignment
- Base / SFT / DPO generation evaluation
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
Rule-based negative candidate generation
    ↓
DPO preference pair construction
    ↓
Qwen2.5-7B QLoRA SFT
    ↓
DPO alignment from SFT adapter
    ↓
Base / SFT / DPO evaluation
    ↓
FastAPI demo
```

## 3. Data

Processed data statistics:

| Split           |   Size |
| --------------- | -----: |
| SFT train       | 11,000 |
| SFT validation  |  1,000 |
| DPO prompt pool |  6,000 |
| DPO train       | 10,758 |
| DPO validation  |  2,689 |
| Dev eval        |    500 |
| Final test      |    269 |

The DPO v1 data contains three types of rejected responses:

* `exaggerated`: exaggerated or forbidden expressions
* `generic`: generic template-like copy
* `weak`: weakened copy with missing selling points

## 4. Training

### SFT

Base model:

```text
Qwen/Qwen2.5-7B-Instruct
```

Training method:

```text
QLoRA / LoRA SFT
```

SFT summary:

| Metric     |     Value |
| ---------- | --------: |
| Epochs     |         2 |
| Steps      |     2,750 |
| Train loss |    2.5848 |
| Runtime    | 7,045.76s |

### DPO

DPO is trained from the SFT adapter using the rule-based DPO v1 preference dataset.

DPO summary:

| Metric     |     Value |
| ---------- | --------: |
| Epochs     |         1 |
| Steps      |       625 |
| Train loss |    0.0156 |
| Runtime    | 3,247.98s |

## 5. Evaluation

Final-test size: 269 samples.

| Model                    | Coverage | Forbidden Count | Repetition Ratio | Avg Length | Rule Score |
| ------------------------ | --------: | --------------: | ---------------: | ---------: | ---------: |
| Base Qwen2.5-7B-Instruct |   0.9260 |          0.0186 |           0.0238 |   191.9963 |     2.4888 |
| SFT                      |   0.8568 |          0.0000 |           0.0140 |    81.9814 |     3.0921 |
| SFT + DPO                |   0.8766 |          0.0000 |           0.0215 |    60.2119 |     3.0177 |

## 6. Interpretation

SFT is the main source of improvement. It improves the overall rule score, removes forbidden expressions, reduces repetition, and shortens the output length.

DPO v1 improves attribute coverage compared with SFT and keeps forbidden expressions at zero, but the overall rule score is slightly lower than SFT. This suggests that rule-based DPO v1 makes the model more concise and conservative. Future work should construct DPO v2 using model-generated candidates and LLM-as-Judge preference labels.

## 7. Repository Structure

```text
app/                     FastAPI demo
configs/                 LLaMA-Factory configs
scripts/                 Training scripts
src/                     Data processing, generation, and evaluation scripts
docs/                    Experiment summary and comparison samples
docs/results/            Evaluation CSV and dataset report
```

## 8. Reproduce Data Processing

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

## 9. Training

```bash
bash scripts/train_sft.sh
bash scripts/train_dpo.sh
```

## 10. Generation and Evaluation

```bash
python src/generate_model_outputs.py \
  --model_type base \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --input_file data/processed/final_test.jsonl \
  --output_file outputs/eval_reports/base_generations.jsonl \
  --max_samples 269

python src/generate_model_outputs.py \
  --model_type sft \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/models/qwen2_5_7b_adgen_sft \
  --input_file data/processed/final_test.jsonl \
  --output_file outputs/eval_reports/sft_generations.jsonl \
  --max_samples 269

python src/generate_model_outputs.py \
  --model_type dpo \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/models/qwen2_5_7b_adgen_dpo \
  --input_file data/processed/final_test.jsonl \
  --output_file outputs/eval_reports/dpo_generations.jsonl \
  --max_samples 269

python src/eval_generation_file.py \
  --generation_file outputs/eval_reports/base_generations.jsonl \
  --output_prefix base

python src/eval_generation_file.py \
  --generation_file outputs/eval_reports/sft_generations.jsonl \
  --output_prefix sft

python src/eval_generation_file.py \
  --generation_file outputs/eval_reports/dpo_generations.jsonl \
  --output_prefix dpo
```

## 11. Limitations

* DPO v1 uses rule-based negative responses rather than human preference labels.
* Rule-based evaluation cannot fully capture ad-copy attractiveness and business conversion.
* DPO v1 improves coverage but does not outperform SFT on the overall rule score.

## 12. Future Work

* Build DPO v2 from model-generated candidates.
* Add LLM-as-Judge preference labeling.
* Add human preference evaluation.
* Support multiple copywriting styles and business constraints.
