# Evaluation Summary

Final-test size: 269 samples.

| Model | Coverage | Forbidden Count | Repetition Ratio | Avg Length | Rule Score |
|---|---:|---:|---:|---:|---:|
| Base Qwen2.5-7B-Instruct | 0.9260 | 0.0186 | 0.0238 | 191.9963 | 2.4888 |
| SFT | 0.8568 | 0.0000 | 0.0140 | 81.9814 | 3.0921 |
| DPO v1 | 0.8766 | 0.0000 | 0.0215 | 60.2119 | 3.0177 |
| DPO v2-small | 0.8652 | 0.0000 | 0.0156 | 79.0855 | 3.1371 |

## Interpretation

SFT is the main source of improvement over the base model. It reduces output length, removes forbidden expressions, lowers repetition, and improves the overall rule score from 2.4888 to 3.0921.

DPO v1 improves attribute coverage compared with SFT, but it produces shorter outputs and slightly higher repetition, leading to a lower overall rule score than SFT.

DPO v2-small uses model-generated candidates and LLM-as-Judge filtering to construct high-confidence preference pairs. It achieves the best overall rule score, 3.1371, while keeping forbidden expressions at zero and maintaining a natural average length close to SFT.
