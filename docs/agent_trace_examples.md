# AdCopy\-ReAct Agent Trace Examples

This document provides trace examples for **AdCopy\-ReAct Agent v3**, a ReAct\-style tool\-using agent for e\-commerce ad copy generation and self\-revision\.

The agent follows:

```text
Thought → Action → Observation → Thought → Action → Observation → Finish
```

### Agent Tools

| Tool                 | Purpose                                                      |
| -------------------- | ------------------------------------------------------------ |
| parse\_attributes    | Parse product attributes from user input                     |
| generate\_candidates | Generate ad\-copy candidates using Qwen2\.5\-7B \+ DPO v2\-small LoRA adapter |
| rule\_evaluate       | Compute coverage, forbidden words, conflicts, repetition, length, and total rule score |
| diagnose\_issues     | Detect missing attributes, unsupported color descriptions, conflicts, length/repetition issues |
| judge\_copy          | Use LLM\-as\-Judge to evaluate coverage, factual consistency, naturalness, attractiveness, and compliance |
| rewrite\_copy        | Use LLM rewrite tool to revise a selected candidate          |
| select\_best         | Select the best candidate based on combined rule \+ judge score |
| finish               | Return final ad copy, scores, and trace                      |

### Example 1: Normal Generation and Selection

#### Input

商品属性：类型=裙；材质=雪纺；颜色=黑色；风格=优雅；图案=印花。

#### Trace File

docs/agent\_traces/adcopy\_agent\_trace\_v3\_normal\.json

#### Flow

```text
parse_attributes
→ generate_candidates
→ rule_evaluate
→ diagnose_issues
→ judge_copy
→ select_best
→ finish
```

The agent generates multiple candidates using the DPO v2\-small model, detects an unsupported color issue in one candidate, avoids selecting it, and returns a clean final copy\.

### Example 2: ReAct Self\-Revision with LLM Rewrite

#### Input

商品属性：类型=裤；版型=显瘦；颜色=黑色；风格=通勤；图案=条纹；裤长=九分裤；裤腰型=高腰；裤款式=直筒。

#### Trace File

docs/agent\_traces/adcopy\_agent\_trace\_v3\_force\_rewrite\_demo\_final\.json

#### Flow

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

The agent first selects the best candidate, then enters a controlled rewrite demo mode to demonstrate self\-revision\. It calls the LLM rewrite tool, re\-evaluates the rewritten candidate, and finally selects the rewritten version\.

#### Key Evidence

- rewrite\_mode = llm\_rewrite

- final source = llm\_rewrite

### Why This Is an Agent Instead of a Fixed Pipeline

A fixed pipeline executes predefined steps once\.

AdCopy\-ReAct Agent maintains state and performs tool calls based on observations:

```text
Perceive product attributes
→ Decide which tool to call
→ Observe tool outputs
→ Diagnose problems
→ Decide whether to revise or stop
→ Re-evaluate after revision
```

This makes it a ReAct\-style tool\-using agent rather than a one\-shot generation pipeline\.

#### Input

商品属性：类型=裤；版型=显瘦；颜色=黑色；风格=通勤；图案=条纹；裤长=九分裤；裤腰型=高腰；裤款式=直筒。

#### Trace File

docs/agent\_traces/adcopy\_agent\_trace\_v3\_force\_rewrite\_demo\_final\.json

#### Flow

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

The agent first selects the best candidate, then enters a controlled rewrite demo mode to demonstrate self\-revision\. It calls the LLM rewrite tool, re\-evaluates the rewritten candidate, and finally selects the rewritten version\.

#### Key Evidence

- **rewrite\_mode** = llm\_rewrite

- **final source** = llm\_rewrite

### Why This Is an Agent Instead of a Fixed Pipeline

A fixed pipeline executes predefined steps once\.

AdCopy\-ReAct Agent maintains state and performs tool calls based on observations:

```text
Perceive product attributes
→ Decide which tool to call
→ Observe tool outputs
→ Diagnose problems
→ Decide whether to revise or stop
→ Re-evaluate after revision
```

This makes it a ReAct\-style tool\-using agent rather than a one\-shot generation pipeline\.

# docs/agent\_summary\.md

## AdCopy\-ReAct Agent v3 Summary

### Overview

AdCopy\-ReAct Agent is an agentic extension of the Qwen2\.5\-based e\-commerce ad copy post\-training system\.

The original system includes:

- SFT training on e\-commerce ad\-copy data

- Rule\-based DPO v1

- Model\-generated \+ LLM\-as\-Judge DPO v2\-small

- Rule\-based automatic evaluation

- Base / SFT / DPO / DPO v2 comparisons

AdCopy\-ReAct Agent further wraps the DPO v2\-small model as a generation tool and adds a ReAct\-style tool\-use loop for evaluation, diagnosis, revision, and final selection\.

### Architecture

```text
User Product Attributes
        ↓
ReAct Agent Controller
        ↓
Thought → Action → Observation Loop
        ↓
Tools:
  - Attribute Parser
  - DPO v2-small Generator
  - Rule Evaluator
  - Issue Diagnoser
  - LLM-as-Judge
  - LLM Rewrite Tool
  - Best Candidate Selector
        ↓
Final Ad Copy + Scores + Trace
```

### Main Components

#### 1\. Generator Tool

The generator tool uses:

Qwen/Qwen2\.5\-7B\-Instruct \+ DPO v2\-small LoRA adapter

It generates multiple candidate ad copies from product attributes\.

#### 2\. Rule Evaluation Tool

The rule evaluator computes:

- Attribute coverage

- Forbidden word count

- Conflict count

- Repetition ratio

- Length score

- Total rule score

#### 3\. Diagnosis Tool

The diagnosis tool detects:

- Missing attributes

- Unsupported color descriptions

- Attribute conflicts

- Forbidden claims

- Overly short or overly long outputs

- High repetition

#### 4\. LLM\-as\-Judge Tool

The judge tool evaluates each candidate on five dimensions:

- Attribute coverage

- Factual consistency

- Naturalness

- Attractiveness

- Compliance

#### 5\. LLM Rewrite Tool

The rewrite tool revises a selected candidate according to:

- Product attributes

- Detected issues

- Judge comments

- Compliance constraints

The rewritten candidate is then re\-evaluated before final selection\.

### ReAct Loop

A typical full self\-revision trace is:

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

### Current Version

AdCopy\-ReAct Agent v3

Implemented abilities:

- Real DPO v2\-small model generation

- Rule\-based observation

- LLM\-as\-Judge semantic observation

- Unsupported color fabrication detection

- LLM rewrite tool

- Rule \+ judge combined scoring

- Self\-revision loop

- Full trace output

### Why It Matters

A normal LLM generates a single output directly\.

AdCopy\-ReAct Agent can:

- Generate multiple candidates

- Evaluate each candidate with tools

- Detect factual or attribute\-level problems

- Avoid selecting problematic candidates

- Rewrite when needed

- Re\-evaluate rewritten output

- Return a final copy with an auditable trace
