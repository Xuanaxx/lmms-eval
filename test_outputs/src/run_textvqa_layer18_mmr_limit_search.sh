#!/bin/bash
set -euo pipefail

source /home/user/czx/uv_env/.tokencompression/bin/activate

export CUDA_VISIBLE_DEVICES=5
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_datasets
export HF_HUB_CACHE=/tmp/hf_hub
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export MPLCONFIGDIR=/tmp/matplotlib

OUT_ROOT=/tmp/textvqa_layer18_mmr_limit_search
mkdir -p "${OUT_ROOT}"

python /home/user/czx/MLLM_Token_Compression_Workdir/src/LLaVA_infer_lmm_evals/textvqa_layer18_mmr_limit_search.py \
  --sample-size 500 \
  --sample-seed 42 \
  --scoring-layer 18 \
  --min-keep-tokens 0 \
  --output-jsonl "${OUT_ROOT}/records.jsonl" \
  --summary-json "${OUT_ROOT}/summary.json" \
  --sample-index-json "${OUT_ROOT}/sample_indices.json"
