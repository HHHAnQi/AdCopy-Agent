# AdCopy\-ReAct Agent v3 Summary

## Overview

AdCopy\-ReAct Agent is an agentic extension of the Qwen2\.5\-based e\-commerce ad copy post\-training system\.

The original system includes:

- SFT training on e\-commerce ad\-copy data

- Rule\-based DPO v1

- Model\-generated \+ LLM\-as\-Judge DPO v2\-small

- Rule\-based automatic evaluation

- Base / SFT / DPO / DPO v2 comparisons

AdCopy\-ReAct Agent further wraps the DPO v2\-small model as a generation tool and adds a ReAct\-style tool\-use loop for evaluation, diagnosis, revision, and final selection\.

---

## Architecture

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

## Main Components

### 1\. Generator Tool

The generator tool uses:

Qwen/Qwen2\.5\-7B\-Instruct \+ DPO v2\-small LoRA adapter

It generates multiple candidate ad copies from product attributes\.

### 2\. Rule Evaluation Tool

The rule evaluator computes:

- Attribute coverage

- Forbidden word count

- Conflict count

- Repetition ratio

- Length score

- Total rule score

### 3\. Diagnosis Tool

The diagnosis tool detects:

- Missing attributes

- Unsupported color descriptions

- Attribute conflicts

- Forbidden claims

- Overly short or overly long outputs

- High repetition

### 4\. LLM\-as\-Judge Tool

The judge tool evaluates each candidate on five dimensions:

- Attribute coverage

- Factual consistency

- Naturalness

- Attractiveness

- Compliance

### 5\. LLM Rewrite Tool

The rewrite tool revises a selected candidate according to:

- Product attributes

- Detected issues

- Judge comments

- Compliance constraints

The rewritten candidate is then re\-evaluated before final selection\.

## ReAct Loop

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

## Current Version

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

## Why It Matters

A normal LLM generates a single output directly\.

AdCopy\-ReAct Agent can:

- Generate multiple candidates

- Evaluate each candidate with tools

- Detect factual or attribute\-level problems

- Avoid selecting problematic candidates

- Rewrite when needed

- Re\-evaluate rewritten output

- Return a final copy with an auditable trace
