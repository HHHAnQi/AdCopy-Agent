#!/usr/bin/env bash
set -euo pipefail

# 注意：
# LLaMA-Factory 默认从其项目目录下的 data/dataset_info.json 读取数据集配置。
# 你需要将本项目的 configs/llamafactory_dataset_info.json 复制到
# LLaMA-Factory 的 data/dataset_info.json，或按你的安装方式调整数据集配置路径。

llamafactory-cli train configs/dpo_qwen2_5_7b_lora.yaml

