#!/bin/bash
set -euo pipefail

source /home/user/czx/uv_env/.tokencompression/bin/activate

export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/home/user/czx/.cache/huggingface
export HF_DATASETS_CACHE=/home/user/czx/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/user/czx/.cache/huggingface/hub
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error

ROOT=/home/user/czx/open_source_proj/lmms-eval
OUT_ROOT=${ROOT}/outputs/logs/textvqa_delete_cost_keep_ratio_layer17
MODEL=/home/user/czx/model/llava-hf/llava-1.5-7b-hf

mkdir -p "${OUT_ROOT}"
cd "${ROOT}"

python3 outputs/src/textvqa_delete_cost_keep_ratio_experiment.py \
  --model-path "${MODEL}" \
  --sample-size 500 \
  --sample-seed 42 \
  --attn-layer 17 \
  --norm-weight 1.0 \
  --combine-type add \
  --max-new-tokens 8 \
  --output-jsonl "${OUT_ROOT}/records.jsonl" \
  --summary-json "${OUT_ROOT}/summary.json" \
  --sample-index-json "${OUT_ROOT}/sample_indices.json"
