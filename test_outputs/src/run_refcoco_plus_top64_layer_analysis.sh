#!/bin/bash
set -euo pipefail

source /home/user/czx/uv_env/.tokencompression/bin/activate

export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/home/user/czx/.cache/huggingface
export HF_HUB_CACHE=/home/user/czx/.cache/huggingface/hub
export HF_DATASETS_CACHE=/home/user/czx/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error

OUT_ROOT=/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/refcoco_plus_lmms_eval_bbox_top64_last_avg_score_distribution
mkdir -p "${OUT_ROOT}"
export MPLCONFIGDIR="${OUT_ROOT}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

python /home/user/czx/open_source_proj/lmms-eval/outputs/src/refcoco_plus_top64_layer_analysis.py \
  --sample-size 1000 \
  --sample-seed 42 \
  --dataset-split val \
  --layers 0-31 \
  --topk 64 \
  --max-new-tokens 32 \
  --caption-metrics Bleu_4,Bleu_3,Bleu_2,Bleu_1,METEOR,ROUGE_L,CIDEr \
  --prompt-source lmms_eval_bbox \
  --attn-anchor last \
  --pool-type avg \
  --store-score-distribution \
  --output-jsonl "${OUT_ROOT}/records.jsonl" \
  --summary-json "${OUT_ROOT}/summary.json" \
  --sample-index-json "${OUT_ROOT}/sample_indices.json"
