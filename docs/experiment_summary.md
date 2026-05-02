# Experiment Summary

## Project

**LLM Post-Training System for E-commerce Ad Copy Generation**

This project builds a complete post-training pipeline for Chinese e-commerce advertisement copy generation. It includes data preprocessing, SFT dataset construction, rule-based DPO preference pair construction, model-generated DPO v2 pair construction, Qwen2.5-7B QLoRA SFT, DPO alignment, automatic evaluation, and a FastAPI demo.

## Data

The project uses AdvertiseGen-style product attribute-to-copy data.

Processed v1 training data:

| Split | Size |
|-------|-----:|
| SFT train | 11,000 |
| SFT validation | 1,000 |
| DPO prompt pool | 6,000 |
| DPO v1 train | 10,758 |
| DPO v1 validation | 2,689 |
| Dev eval | 500 |
| Final test | 269 |

DPO v1 uses rule-based rejected responses:

- **exaggerated**: contains exaggerated or forbidden expressions.
- **generic**: generic template-like copy.
- **weak**: weakened copy with missing or reduced selling points.

## DPO v2-small Data

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

## Pair Quality Statistics on the DPO v2-small Training Split

| Metric | Value |
|--------|------:|
| Avg score gap | 0.7196 |
| Avg chosen rule score | 3.2285 |
| Avg rejected rule score | 2.6699 |
| Avg chosen judge score | 23.7492 |
| Avg rejected judge score | 19.2260 |
| Judge chosen > rejected ratio | 1.0000 |
| Rule chosen > rejected ratio | 1.0000 |

## Training

### SFT

Base model:

Qwen/Qwen2.5-7B-Instruct

Method:

QLoRA / LoRA SFT

SFT training summary:

| Metric | Value |
|--------|------:|
| Epochs | 2 |
| Steps | 2,750 |
| Train loss | 2.5848 |
| Runtime | 7,045.76s |
| Train samples / second | 3.122 |

### DPO v1

DPO v1 is initialized from the SFT adapter and trained on the rule-based preference dataset.

DPO v1 training summary:

| Metric | Value |
|--------|------:|
| Epochs | 1 |
| Steps | 625 |
| Train loss | 0.0156 |
| Runtime | 3,247.98s |

### DPO v2-small

DPO v2-small is initialized from the SFT adapter and trained on high-confidence model-generated preference pairs.

DPO v2-small training summary:

| Metric | Value |
|--------|------:|
| Epochs | 1 |
| Steps | 41 |
| Train loss | 1.8015 |
| Runtime | 230.31s |
| Train samples / second | 1.402 |

Although DPO v2-small has a weaker reward-accuracy signal during training, final generation evaluation shows that it improves the output distribution and obtains the best overall rule score.

## Evaluation

Final-test size: 269 samples.

| Model | Coverage | Forbidden Count | Repetition Ratio | Avg Length | Rule Score |
|-------|---------:|----------------:|-----------------:|-----------:|------------:|
| Base Qwen2.5-7B-Instruct | 0.9260 | 0.0186 | 0.0238 | 191.9963 | 2.4888 |
| SFT | 0.8568 | 0.0000 | 0.0140 | 81.9814 | 3.0921 |
| DPO v1 | 0.8766 | 0.0000 | 0.0215 | 60.2119 | 3.0177 |
| DPO v2-small | 0.8652 | 0.0000 | 0.0156 | 79.0855 | 3.1371 |

## Main Findings

- SFT is the main improvement over the base model.  
  SFT improves the rule score from 2.4888 to 3.0921, removes forbidden expressions, reduces repetition, and shortens outputs from 192 characters to 82 characters on average.

- DPO v1 improves coverage but over-compresses generation.  
  DPO v1 improves coverage to 0.8766 but reduces average length to 60.21 and has a slightly lower rule score than SFT.

- DPO v2-small achieves the best overall rule score.  
  DPO v2-small uses model-generated candidates and LLM-as-Judge high-confidence filtering. It achieves the best rule score, 3.1371, while keeping forbidden count at zero and maintaining a natural average length of 79.09.

## Limitations

- DPO v2-small uses only 379 high-confidence preference pairs.
- LLM-as-Judge is still an automatic evaluator and may contain bias.
- Rule-based metrics cannot fully capture copywriting attractiveness or real business conversion.
- Human preference evaluation is not yet included.

## Future Work

- Scale DPO v2 to 2,000–3,000 prompts.
- Add human preference evaluation.
- Improve LLM-as-Judge prompts for stricter factual consistency.
- Support multiple copywriting styles such as premium, playful, concise, and Xiaohongshu-style copy.
