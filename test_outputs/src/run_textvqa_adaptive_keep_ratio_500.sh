#!/bin/bash
set -euo pipefail

source /home/user/czx/uv_env/.tokencompression/bin/activate

export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/home/user/czx/.cache/huggingface
export HF_DATASETS_CACHE=/home/user/czx/.cache/huggingface/datasets
export HF_HUB_CACHE=/home/user/czx/.cache/huggingface/hub
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_USE_PYTORCH_OPERATORS=1
export TRANSFORMERS_VERBOSITY=error

ROOT=/home/user/czx/open_source_proj/lmms-eval
OUT_ROOT=${ROOT}/outputs/logs/adaptive_textvqa_runs
MODEL=/data1/czx/MLLM_Token_Compression_Workdir/output/llava_pre25_iqr15_19_mmr_flexattnv2

mkdir -p "${OUT_ROOT}"
cd "${ROOT}"

run_eval() {
  local run_name="$1"
  local model_args="$2"

  rm -f "${OUT_ROOT}/${run_name}_metrics.jsonl"
  accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_hf \
    --model_args "${model_args},adaptive_metrics_log_path=${OUT_ROOT}/${run_name}_metrics.jsonl" \
    --tasks textvqa_val \
    --limit 500 \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${run_name}" \
    --output_path "${OUT_ROOT}/${run_name}"
}

COMMON_ARGS="pretrained=${MODEL},device_map=auto,attn_implementation=eager"

run_eval "fixed_0125" "${COMMON_ARGS},prefill_keep_ratio=0.125,adaptive_keep_ratio_mode=none"
run_eval "fixed_0250" "${COMMON_ARGS},prefill_keep_ratio=0.25,adaptive_keep_ratio_mode=none"
run_eval "adaptive_attn_entropy" "${COMMON_ARGS},prefill_keep_ratio=0.25,adaptive_keep_ratio_mode=attn_entropy_linear,adaptive_keep_ratio_min=0.125,adaptive_keep_ratio_max=0.30,adaptive_keep_ratio_topk_ratio=0.125,adaptive_keep_ratio_attention_weight=0.35,adaptive_keep_ratio_entropy_weight=0.65"
run_eval "adaptive_hybrid" "${COMMON_ARGS},prefill_keep_ratio=0.25,adaptive_keep_ratio_mode=hybrid_linear,adaptive_keep_ratio_min=0.125,adaptive_keep_ratio_max=0.30,adaptive_keep_ratio_topk_ratio=0.125,adaptive_keep_ratio_attention_weight=0.2,adaptive_keep_ratio_entropy_weight=0.4,adaptive_keep_ratio_coverage_weight=0.4"
