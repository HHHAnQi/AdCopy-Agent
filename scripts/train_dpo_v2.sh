#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

llamafactory-cli train configs/dpo_v2_qwen2_5_7b_lora.yaml
