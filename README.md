# 电商广告文案生成与偏好对齐系统

基于 Qwen2.5 + LoRA + Rule-based DPO 的中文电商广告生成项目，覆盖数据清洗、SFT 数据构造、DPO 偏好对齐样本构造、规则评估、训练配置与 FastAPI 服务 Demo。

## 项目阶段说明

### 阶段一：本地工程与数据构造阶段（已完成）

- AdvertiseGen 数据解析
- SFT 数据构造
- DPO prompt pool 构造
- rule-based negative candidates
- DPO pair 构造与过滤
- `dataset_report`
- mock eval
- FastAPI demo

### 阶段二：模型后训练阶段（待 GPU 环境）

- Qwen2.5-7B-Instruct QLoRA SFT
- 使用 SFT 模型生成真实 candidates
- 构造 model-generated DPO v2
- DPO 偏好对齐
- Base/SFT/DPO 三模型评估

## 1) 项目背景与业务价值

电商广告文案通常要求同时满足「卖点突出、表达自然、避免违规、属性一致」四类目标。纯 SFT 往往能学到语气与结构，但在事实一致性、违规词控制和偏好对齐上存在不足。本项目通过 **SFT + DPO** 两阶段流程，提升文案质量与可控性：

- 提高属性覆盖率，减少“写偏题”；
- 降低违规夸大表达；
- 利用偏好对比样本强化“好文案 > 差文案”的排序能力；
- 提供可复现的数据管线和可部署 API 入口。

## 2) 数据集说明

- 数据集：`shibing624/AdvertiseGen`
- 本地路径：
  - `data/AdvertiseGen/train.json`
  - `data/AdvertiseGen/dev.json`
- 常见字段：
  - `content`：商品属性 kv 串（例如 `类型#裙*风格#通勤*图案#纯色`）
  - `summary`：广告文案

## 3) 数据处理流程

完整链路如下（按执行顺序）：

1. `src/parse_kv.py`  
   解析 AdvertiseGen 的 `content` 字段，生成结构化属性并格式化为自然语言输入。

2. `src/prepare_sft.py`  
   完成清洗、去重、数据划分，生成：
   - `sft_train.jsonl`
   - `sft_val.jsonl`
   - `dpo_prompt_pool.jsonl`
   - `dev_eval.jsonl`
   - `final_test.jsonl`

3. `src/generate_candidates.py`  
   从 `dpo_prompt_pool.jsonl` 为每个 prompt 生成候选文案（支持 `mock` / `transformers`）。

4. `src/build_dpo_pairs.py`  
   用 `rule_score` 对候选打分，构造 DPO pair（训练版 + 分析版）。

5. `src/inspect_dpo.py`  
   抽样检查 DPO 数据质量，查看 source 分布与异常样本。

6. `src/dataset_report.py`  
   生成全项目数据统计报告（JSON + Markdown）。

## 4) 当前数据规模统计

- `sft_train`: 11000
- `sft_val`: 1000
- `dpo_prompt_pool`: 6000
- `dpo_train`: 10540
- `dpo_val`: 2635
- `dev_eval`: 500
- `final_test`: 500
- `DPO meta pairs`: 13175
- `SFT output 平均长度`: 109.31
- `DPO chosen_score avg`: 3.1729
- `DPO rejected_score avg`: 2.1538
- `chosen_score <= rejected_score`: 0

> 说明：当前评估结果来自 `mock eval` 流程，用于验证工程链路与数据质量，不代表真实模型训练后的线上效果。

## 5) SFT 数据格式示例

```json
{
  "instruction": "请根据商品属性生成一段中文电商广告文案，要求突出核心卖点，表达自然，有购买吸引力，不得编造未提供的信息。",
  "input": "商品属性：类型=上衣；材质=牛仔布；颜色=白色；风格=简约。",
  "output": "白色牛仔上衣清爽百搭，简约风格耐看不挑人，日常通勤都能轻松驾驭。"
}
```

## 6) DPO 数据格式示例

训练版（用于 LLaMA-Factory）：

```json
{
  "instruction": "请根据商品属性生成一段中文电商广告文案，要求突出核心卖点，表达自然，有购买吸引力，不得编造未提供的信息。",
  "input": "商品属性：类型=上衣；材质=牛仔布；颜色=白色；风格=简约。",
  "chosen": "白色牛仔上衣清爽百搭，简约气质自然耐看，面料挺括有型，日常穿搭省心。",
  "rejected": "这款产品全网第一，效果100%，必买神级单品，品质永久 guaranteed。"
}
```

分析版（用于统计分析）在 `outputs/dpo/dpo_pairs_with_meta.jsonl`，额外包含：
- `chosen_source`
- `rejected_source`
- `chosen_score`
- `rejected_score`

## 7) DPO 构造策略（当前版本）

- `chosen`：优先 `gold`，且需满足无禁用词、无属性冲突、长度合格；
- `rejected`：来自 `exaggerated / generic / weak`；
- 严格过滤：
  - `chosen_score <= rejected_score` 直接丢弃；
  - `chosen_score - rejected_score < 0.5` 丢弃；
  - `chosen == rejected` 丢弃；
  - `chosen` 含禁用词丢弃；
- 目标：保证偏好方向明确，降低噪声 pair。

## 8) 本地运行命令

### 数据准备

```bash
python src/prepare_sft.py
python src/generate_candidates.py --mode mock --input_file data/processed/dpo_prompt_pool.jsonl --output_file outputs/candidates/candidates.jsonl
python src/build_dpo_pairs.py
python src/inspect_dpo.py --input_file outputs/dpo/dpo_pairs_with_meta.jsonl --num_samples 10
python src/dataset_report.py
```

### 评估

```bash
python src/run_eval.py --mode mock --input_file data/processed/final_test.jsonl
```

> 注意：这里是 `mock eval`，用于流程联调与指标检查，不代表真实 Base/SFT/DPO 模型效果。

## 9) 训练配置说明

- SFT 配置：`configs/sft_qwen2_5_7b_lora.yaml`
- DPO 配置：`configs/dpo_qwen2_5_7b_lora.yaml`
- 训练脚本：
  - `scripts/train_sft.sh`
  - `scripts/train_dpo.sh`
- 数据集映射：`configs/llamafactory_dataset_info.json`

> 注意：LLaMA-Factory 默认读取其目录下的 `data/dataset_info.json`。请将本项目的 `configs/llamafactory_dataset_info.json` 复制到对应位置，或按实际安装方式调整。

## 10) FastAPI Demo 使用方式

启动：

```bash
uvicorn app.main:app --reload --port 8000
```

健康检查：

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

生成文案：

```bash
curl -X POST "http://127.0.0.1:8000/generate_ad" \
  -H "Content-Type: application/json" \
  -d '{
    "attributes": {
      "类型": "上衣",
      "材质": "牛仔布",
      "颜色": "白色",
      "风格": "简约"
    },
    "num_candidates": 3
  }'
```

## 11) 当前限制

- 当前 DPO 方案是 **rule-based DPO v1**，打分与偏好构造主要依赖规则；
- 候选质量仍受 mock/基础生成策略影响，真实上限受候选生成模型能力约束；
- 事实一致性冲突检测为关键词规则，后续可进一步升级为语义级校验。

## 12) 后续升级方向

- 使用 SFT 模型生成更高质量候选（替代纯规则扰动）；
- 引入 **LLM-as-Judge** 做偏好打分，结合规则形成混合评估；
- 加入多维 reward（真实性、可读性、转化导向、品牌语气）；
- 形成线上 A/B 评估闭环。

## 13) 简历写法建议

可参考以下表述：

- 负责搭建“电商广告文案生成与偏好对齐系统”，基于 Qwen2.5 + LoRA 实现 SFT 与 DPO 两阶段训练流程。
- 从零构建 AdvertiseGen 数据管线，完成清洗、去重、SFT/DPO 数据构造与自动评估报告，沉淀可复现脚本体系。
- 设计 rule-based 偏好构造策略（事实冲突、违规词、覆盖率、长度、重复度等），产出万级高质量 DPO 样本并实现严格质量过滤。
- 提供 FastAPI 推理服务 Demo，支持候选生成、规则打分排序和最佳文案返回，具备工程落地能力。

