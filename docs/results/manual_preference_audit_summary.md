# Manual Preference Audit Summary

## Setup

- Comparison: **sft** vs **dpo_v2**
- Number of audited examples: **50**
- Annotator: project author
- Criteria: attribute coverage, factual consistency, fluency, attractiveness, and compliance

## Results

| Winner | Count | Rate |
|---|---:|---:|
| sft | 9 | 18.0% |
| dpo_v2 | 28 | 56.0% |
| tie | 13 | 26.0% |

## Key Metric

- **dpo_v2 win rate:** 56.0%
- **dpo_v2 win-or-tie rate:** 82.0%

## Interpretation

In this 50-sample manual preference audit, **dpo_v2** was preferred over **sft** in **56.0%** of cases and was at least tied in **82.0%** of cases. This provides human preference evidence that DPO v2-small improves perceived generation quality beyond automatic rule-based metrics.

## Notes

- This is a small-scale manual audit intended as a qualitative validation complement.
- The rule-based score is used for engineering diagnostics and does not fully replace human preference evaluation.
- The audit reasons are stored in `manual_preference_audit.csv`.
