# Evaluation Summary

Final-test size: 269 samples.

| Model | Coverage | Forbidden Count | Repetition Ratio | Avg Length | Rule Score |
|---|---:|---:|---:|---:|---:|
| Base Qwen2.5-7B-Instruct | 0.9260 | 0.0186 | 0.0238 | 191.9963 | 2.4888 |
| SFT | 0.8568 | 0.0000 | 0.0140 | 81.9814 | 3.0921 |
| SFT + DPO | 0.8766 | 0.0000 | 0.0215 | 60.2119 | 3.0177 |

## Interpretation

SFT is the main source of overall rule-score improvement. Compared with the base model, SFT reduces output length, removes forbidden expressions, lowers repetition, and improves the overall rule score.

DPO v1 keeps forbidden count at zero and improves attribute coverage compared with SFT, but its overall rule score is slightly lower than SFT. This suggests that the current rule-based DPO preference data makes the model more concise and coverage-aware, but also more conservative. A future DPO v2 should use model-generated candidates and LLM-as-Judge preference labels.
