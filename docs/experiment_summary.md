# Experiment Summary

## Project

**LLM Post-Training System for E-commerce Ad Copy Generation**

This project builds a complete post-training pipeline for Chinese e-commerce advertisement copy generation. It includes data preprocessing, SFT dataset construction, rule-based DPO preference pair construction, Qwen2.5-7B QLoRA SFT, DPO alignment, automatic evaluation, and a FastAPI demo.

## Data

The project uses AdvertiseGen-style product attribute-to-copy data.

Processed training data:

| Split | Size |
|---|---:|
| SFT train | 11,000 |
| SFT validation | 1,000 |
| DPO prompt pool | 6,000 |
| DPO train | 10,758 |
| DPO validation | 2,689 |
| Dev eval | 500 |
| Final test | 269 |

The DPO v1 dataset is constructed with a gold chosen response and three types of rejected responses:

- **exaggerated**: contains exaggerated or forbidden expressions.
- **generic**: generic template-like copy.
- **weak**: weakened copy with missing or reduced selling points.

Preference pairs are filtered to ensure that the chosen response has a higher rule score than the rejected response.

## Training

### SFT

Base model:

- Qwen2.5-7B-Instruct

Method:

- QLoRA / LoRA SFT

SFT training summary:

| Metric | Value |
|---|---:|
| Epochs | 2 |
| Steps | 2,750 |
| Train loss | 2.5848 |
| Runtime | 7,045.76s |
| Train samples / second | 3.122 |

### DPO

DPO is initialized from the SFT adapter and trained on the rule-based DPO v1 preference dataset.

DPO training summary:

| Metric | Value |
|---|---:|
| Epochs | 1 |
| Steps | 625 |
| Train loss | 0.0156 |
| Runtime | 3,247.98s |
| Train samples / second | 1.539 |

The DPO reward accuracy becomes very high during training, indicating that the rule-based preference pairs are easy to distinguish. This is useful for a first version of DPO alignment, but future work should include model-generated candidates and LLM-as-Judge labels.

## Evaluation

Final-test size: 269 samples.

| Model | Coverage | Forbidden Count | Repetition Ratio | Avg Length | Rule Score |
|---|---:|---:|---:|---:|---:|
| Base Qwen2.5-7B-Instruct | 0.9260 | 0.0186 | 0.0238 | 191.9963 | 2.4888 |
| SFT | 0.8568 | 0.0000 | 0.0140 | 81.9814 | 3.0921 |
| SFT + DPO | 0.8766 | 0.0000 | 0.0215 | 60.2119 | 3.0177 |

## Main Findings

1. **SFT is the main improvement source.**  
   SFT improves the rule score from 2.4888 to 3.0921, removes forbidden expressions, reduces repetition, and shortens outputs from 192 characters to 82 characters on average.

2. **DPO v1 improves coverage but is more conservative.**  
   Compared with SFT, DPO improves coverage from 0.8568 to 0.8766 and keeps forbidden count at zero, but the overall rule score slightly decreases from 3.0921 to 3.0177.

3. **Rule-based DPO v1 is useful but limited.**  
   The DPO pairs are clean and easy to distinguish, but the preference signal is still rule-based. Future DPO v2 should use SFT-generated candidates, rule filtering, and LLM-as-Judge preference labels.

## Limitations

- The current DPO data is rule-based rather than human-annotated.
- Automatic rule metrics cannot fully measure copywriting quality, attractiveness, or business conversion.
- The final-test set contains 269 samples after filtering.
- DPO v1 does not outperform SFT on overall rule score, though it improves coverage and keeps outputs concise.

## Future Work

- Construct DPO v2 from model-generated candidates.
- Add LLM-as-Judge evaluation for naturalness, attribute consistency, attractiveness, and compliance.
- Add human preference evaluation on a small sample set.
- Support multi-style ad copy generation, such as premium, playful, concise, and Xiaohongshu-style copy.
