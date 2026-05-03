# LLM Post\-Training System for E\-commerce Ad Copy Generation

This project builds a complete LLM post\-training and agentic optimization system for Chinese e\-commerce ad copy generation\.

It starts from supervised fine\-tuning and preference alignment, then further extends the trained model into a ReAct\-style tool\-using agent that can generate, evaluate, diagnose, revise, and select ad copy candidates\.

---

## 1\. Project Overview

The task is to generate attractive and faithful Chinese e\-commerce advertising copy from structured product attributes\.

#### Example input:

```text
商品属性：类型=裙；材质=雪纺；颜色=黑色；风格=优雅；图案=印花。
```

#### Expected output:

这款裙子采用了黑色的底色，搭配精致的印花设计，优雅大方又不失时尚感。雪纺面料轻盈飘逸，穿着舒适自然，轻松展现女性温柔气质。

The system focuses on:

- Attribute coverage

- Factual consistency

- Natural expression

- Marketing attractiveness

- Compliance and avoidance of exaggerated claims

- Automatic evaluation and revision

## 2\. Main Contributions

This project contains four major parts:

### SFT training

Fine\-tunes Qwen2\.5\-7B\-Instruct on e\-commerce ad copy data\.

### DPO v1

Constructs rule\-based preference pairs with exaggerated, generic, and weak rejected responses\.

### DPO v2\-small

Uses SFT\-generated candidates and LLM\-as\-Judge to construct high\-confidence model\-generated preference pairs\.

### AdCopy\-ReAct Agent v3

Wraps the DPO v2\-small model as a generation tool\.

Uses rule\-based evaluation, issue diagnosis, LLM\-as\-Judge, and LLM rewrite tools\.

Follows a ReAct\-style Thought → Action → Observation loop\.

## 3\. System Architecture

```text
AdvertiseGen Dataset
        ↓
Data Cleaning & Prompt Construction
        ↓
SFT Data / DPO Prompt Pool / Final Test
        ↓
Qwen2.5-7B QLoRA SFT
        ↓
Rule-based DPO v1
        ↓
Model-generated + LLM-as-Judge DPO v2-small
        ↓
Base / SFT / DPO / DPO v2 Evaluation
        ↓
AdCopy-ReAct Agent
        ↓
Generate → Evaluate → Diagnose → Judge → Rewrite → Select
```

## 4\. Data Pipeline

The raw data follows the AdvertiseGen\-style format:

```json
{
  "content": "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤",
  "summary": "宽松的阔腿裤这两年真的吸粉不少..."
}
```

The project converts it into instruction\-tuning format:

```json
{
  "instruction": "请根据商品属性生成一段中文电商广告文案，要求突出核心卖点，表达自然，有购买吸引力，不得编造未提供的信息。",
  "input": "商品属性：类型=裤；版型=宽松；风格=性感；图案=线条；裤型=阔腿裤。",
  "output": "宽松的阔腿裤这两年真的吸粉不少..."
}
```

#### Current data scale:

| Split                    | Size   |
| ------------------------ | ------ |
| SFT train                | 11,000 |
| SFT validation           | 1,000  |
| DPO prompt pool          | 6,000  |
| DPO v1 train             | 10,758 |
| DPO v1 validation        | 2,689  |
| DPO v2\-small train      | 323    |
| DPO v2\-small validation | 56     |
| Final test               | 269    |

## 5\. Model Training

### Base Model

Qwen/Qwen2\.5\-7B\-Instruct

### Training Method

The project uses LoRA / QLoRA training with LLaMA\-Factory\.

#### Main training stages:

```text
Base Qwen2.5-7B-Instruct
        ↓
SFT
        ↓
DPO v1
        ↓
DPO v2-small
```

## 6\. DPO v1: Rule\-based Preference Alignment

DPO v1 constructs rejected responses through rule\-based perturbations\.

#### Rejected response types:

| Type        | Description                                              |
| ----------- | -------------------------------------------------------- |
| exaggerated | Adds exaggerated or non\-compliant marketing expressions |
| generic     | Creates vague and generic ad copy                        |
| weak        | Removes or weakens product attribute coverage            |

DPO v1 helps improve compliance and attribute awareness, but may make the model overly conservative and shorten outputs\.

## 7\. DPO v2\-small: Model\-generated \+ LLM\-as\-Judge Preference Pairs

DPO v2\-small addresses the limitation of rule\-based rejected samples\.

#### Pipeline:

```text
DPO prompt pool
        ↓
SFT model generates 5 candidates per prompt
        ↓
Rule scoring
        ↓
LLM-as-Judge scoring
        ↓
High-confidence chosen / rejected pair selection
        ↓
DPO v2-small training
```

#### DPO v2\-small statistics:

| Item                                | Value   |
| ----------------------------------- | ------- |
| Input prompts                       | 1,000   |
| Candidates per prompt               | 5       |
| Total candidates                    | 5,000   |
| Prompts with high\-confidence pairs | 379     |
| Total pairs                         | 379     |
| Train pairs                         | 323     |
| Validation pairs                    | 56      |
| Average score gap                   | 0\.7196 |
| Rule chosen \&gt; rejected ratio    | 1\.0000 |
| Judge chosen \&gt; rejected ratio   | 1\.0000 |

## 8\. Evaluation Metrics

The project uses an automatic rule\-based evaluation script\.

#### Metrics:

| Metric                 | Description                                    |
| ---------------------- | ---------------------------------------------- |
| avg\_coverage          | Attribute coverage score                       |
| avg\_forbidden\_count  | Number of exaggerated or forbidden expressions |
| avg\_repetition\_ratio | Repetition ratio                               |
| avg\_length            | Average output length                          |
| avg\_total\_score      | Combined rule\-based quality score             |

## 9\. Main Results

Final test size: 269\.

| Model                       | Coverage | Forbidden | Repetition | Avg Length | Rule Score |
| --------------------------- | -------- | --------- | ---------- | ---------- | ---------- |
| Base Qwen2\.5\-7B\-Instruct | 0\.9260  | 0\.0186   | 0\.0238    | 191\.9963  | 2\.4888    |
| SFT                         | 0\.8568  | 0\.0000   | 0\.0140    | 81\.9814   | 3\.0921    |
| DPO v1                      | 0\.8766  | 0\.0000   | 0\.0215    | 60\.2119   | 3\.0177    |
| DPO v2\-small               | 0\.8652  | 0\.0000   | 0\.0156    | 79\.0855   | 3\.1371    |

#### Key observations:

- SFT provides the main improvement over the base model\.

- DPO v1 improves coverage but tends to produce shorter outputs\.

- DPO v2\-small achieves the best overall rule score\.

- All post\-trained models reduce forbidden expressions to zero\.

## 10\. AdCopy\-ReAct Agent v3

The project further extends the post\-trained model into a ReAct\-style tool\-using agent\.

### Motivation

A normal LLM generates one answer directly:

```text
Input product attributes → Generate one ad copy → Finish
```

AdCopy\-ReAct Agent performs multi\-step reasoning and tool use:

```text
Perceive product attributes
→ Generate multiple candidates
→ Evaluate candidates
→ Diagnose issues
→ Judge semantic quality
→ Rewrite if needed
→ Re-evaluate
→ Select final copy
```

### Agent Tools

| Tool                 | Purpose                                                      |
| -------------------- | ------------------------------------------------------------ |
| parse\_attributes    | Parse product attributes from user input                     |
| generate\_candidates | Generate candidates using Qwen2\.5 \+ DPO v2\-small adapter  |
| rule\_evaluate       | Compute rule\-based scores                                   |
| diagnose\_issues     | Detect missing attributes, unsupported colors, conflicts, and length issues |
| judge\_copy          | Use LLM\-as\-Judge to score semantic quality                 |
| rewrite\_copy        | Use LLM rewrite tool to revise selected candidate            |
| select\_best         | Select the best candidate using rule \+ judge score          |
| finish               | Return final copy, scores, and trace                         |

### ReAct Loop

Thought → Action → Observation → Thought → Action → Observation → Finish

A full self\-revision trace:

```text
parse_attributes
→ generate_candidates
→ rule_evaluate
→ diagnose_issues
→ judge_copy
→ select_best
→ rewrite_copy
→ rule_evaluate
→ diagnose_issues
→ judge_copy
→ select_best
→ finish
```

## 11\. Agent Trace Examples

Two representative trace examples are provided:

- docs/agent\_trace\_examples\.md

- docs/agent\_summary\.md

- docs/agent\_traces/adcopy\_agent\_trace\_v3\_normal\.json

- docs/agent\_traces/adcopy\_agent\_trace\_v3\_force\_rewrite\_demo\_final\.json

Example 1: normal generation and selection\.

Example 2: controlled self\-revision demo with LLM rewrite\.

In the self\-revision example, the agent successfully calls:

**rewrite\_copy**

and the rewrite mode is:

**llm\_rewrite**

The final selected candidate comes from:

**source = llm\_rewrite**

This demonstrates that the agent can observe, revise, re\-evaluate, and select\.

## 12\. Directory Structure

```text
.
├── configs/
│   ├── dpo_qwen2_5_7b_lora.yaml
│   └── dpo_v2_qwen2_5_7b_lora.yaml
├── data/
│   └── processed/
├── docs/
│   ├── agent_summary.md
│   ├── agent_trace_examples.md
│   ├── agent_traces/
│   ├── comparison_samples.md
│   ├── comparison_samples_v2.md
│   ├── experiment_summary.md
│   └── results/
├── scripts/
│   ├── train_dpo_v2.sh
│   └── download_model.py
├── src/
│   ├── agent/
│   │   ├── schemas.py
│   │   ├── tools.py
│   │   ├── reasoner.py
│   │   ├── workflow.py
│   │   ├── model_generator.py
│   │   ├── judge_tool.py
│   │   └── rewrite_tool.py
│   ├── run_adcopy_agent.py
│   ├── generate_model_outputs.py
│   ├── eval_generation_file.py
│   ├── generate_dpo_v2_candidates.py
│   ├── score_dpo_v2_candidates.py
│   ├── judge_dpo_v2_candidates.py
│   ├── build_dpo_v2_pairs.py
│   └── compare_generations_v2.py
└── README.md
```

## 13\. Quick Start

### 13\.1 Generate outputs from SFT / DPO models

```bash
python src/generate_model_outputs.py \
  --model_type dpo \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/models/qwen2_5_7b_adgen_dpo \
  --input_file data/processed/final_test.jsonl \
  --output_file outputs/eval_reports/dpo_generations.jsonl \
  --max_samples 269 \
  --max_new_tokens 180
```

### 13\.2 Evaluate generated outputs

```bash
python src/eval_generation_file.py \
  --generation_file outputs/eval_reports/dpo_generations.jsonl \
  --output_prefix dpo
```

### 13\.3 Run AdCopy\-ReAct Agent

#### Normal generation and selection:

```bash
export OMP_NUM_THREADS=4

python src/run_adcopy_agent.py \
  --generator_mode model \
  --enable_judge \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/models/qwen2_5_7b_adgen_dpo_v2 \
  --input "商品属性：类型=裙；材质=雪纺；颜色=黑色；风格=优雅；图案=印花。" \
  --output_file outputs/agent/adcopy_agent_trace_v3_normal.json \
  --max_iterations 14
```

#### Self\-revision demo:

```bash
export OMP_NUM_THREADS=4

python src/run_adcopy_agent.py \
  --generator_mode model \
  --enable_judge \
  --force_rewrite_demo \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/models/qwen2_5_7b_adgen_dpo_v2 \
  --input "商品属性：类型=裤；版型=显瘦；颜色=黑色；风格=通勤；图案=条纹；裤长=九分裤；裤腰型=高腰；裤款式=直筒。" \
  --output_file outputs/agent/adcopy_agent_trace_v3_force_rewrite_demo_final.json \
  --max_iterations 16
```

## 14\. Key Files

| File                                  | Description                                        |
| ------------------------------------- | -------------------------------------------------- |
| src/generate\_model\_outputs\.py      | Generate model outputs for Base / SFT / DPO models |
| src/eval\_generation\_file\.py        | Evaluate generation files                          |
| src/generate\_dpo\_v2\_candidates\.py | Generate candidates for DPO v2                     |
| src/judge\_dpo\_v2\_candidates\.py    | LLM\-as\-Judge scoring for DPO v2 data             |
| src/build\_dpo\_v2\_pairs\.py         | Build high\-confidence DPO v2 preference pairs     |
| src/run\_adcopy\_agent\.py            | Run AdCopy\-ReAct Agent                            |
| src/agent/workflow\.py                | ReAct agent main loop                              |
| src/agent/reasoner\.py                | Rule\-based ReAct reasoner                         |
| src/agent/model\_generator\.py        | DPO v2 model generator tool                        |
| src/agent/judge\_tool\.py             | LLM\-as\-Judge tool                                |
| src/agent/rewrite\_tool\.py           | LLM rewrite tool                                   |
| docs/agent\_trace\_examples\.md       | Agent trace examples                               |
| docs/agent\_summary\.md               | Agent summary                                      |

## 15\. Project Status

Current version:

**AdCopy\-ReAct Agent v3**

#### Completed:

- SFT post\-training

- Rule\-based DPO v1

- Model\-generated \+ LLM\-as\-Judge DPO v2\-small

- Base / SFT / DPO / DPO v2 evaluation

- ReAct\-style agent controller

- DPO v2 model generation tool

- Rule evaluation tool

- Issue diagnosis tool

- LLM\-as\-Judge tool

- LLM rewrite tool

- Self\-revision trace

- GitHub\-ready documentation

## 16\. Limitations

#### Current limitations:

- DPO v2\-small uses a small number of high\-confidence preference pairs\.

- Rule\-based metrics are useful for engineering iteration but do not fully replace human evaluation\.

- LLM\-as\-Judge may still have bias\.

- The agent currently targets a vertical ad\-copy generation task rather than a general\-purpose agent\.

- The self\-revision demo uses a controlled rewrite trigger to demonstrate the full loop\.

## 17\. Future Work

#### Planned improvements:

- Expand DPO v2 from 1,000 prompts to a larger candidate pool\.

- Add human preference evaluation for SFT vs DPO v2 and agent rewrite outputs\.

- Improve factual consistency diagnosis beyond rule\-based color and attribute checks\.

- Add a FastAPI endpoint for agent inference\.

- Convert the current Python ReAct workflow into a LangGraph\-based implementation\.

- Add a lightweight web demo for interactive ad\-copy generation and revision\.

## 18\. Project Positioning

This project demonstrates:

- LLM post\-training

- SFT and DPO preference alignment

- Model\-generated preference data construction

- LLM\-as\-Judge evaluation

- Automatic rule\-based evaluation

- ReAct\-style agent design

- Tool use, observation, self\-revision, and traceability

It is designed as a practical LLM application and post\-training project for e\-commerce content generation\.