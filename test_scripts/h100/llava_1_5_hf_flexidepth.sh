#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  conda deactivate || true
fi
source /home/user/czx/uv_env/.tokencompression/bin/activate

cd /home/user/czx/open_source_proj/lmms-eval
python -c "import transformers; print(transformers.__file__)"

export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
export TRANSFORMERS_USE_PYTORCH_OPERATORS="${TRANSFORMERS_USE_PYTORCH_OPERATORS:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-YP2RHFk6AQGUWtgCaMG9hdm685ZtkcFX1Uf5vplQjzI1VHCc}"
export OPENAI_API_URL="${OPENAI_API_URL:-https://xiaoai.plus/v1/chat/completions}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL_PATH="${MODEL_PATH:-/data1/czx/MLLM_Token_Compression_Workdir/output/llava_pre25_iqr15_19_mmr_flexattnv2}"
FLEXIDEPTH_MODEL_FILE="${FLEXIDEPTH_MODEL_FILE:-/home/user/czx/MLLM_Token_Compression_Workdir/src/LLaVA_flexidepth/pre_token_compression_post_flexidepth/modeling_llava_flexidepth_pre_visual_post_all_adaptive_prefill_one_stage_mmr.py}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
FLEXIDEPTH_LAYERS="${FLEXIDEPTH_LAYERS:-20-31:all}"
FLEXIDEPTH_TAU="${FLEXIDEPTH_TAU:-0.5}"
FLEXIDEPTH_ROUTER_RATIO="${FLEXIDEPTH_ROUTER_RATIO:-16}"
FLEXIDEPTH_ADAPTER_RATIO="${FLEXIDEPTH_ADAPTER_RATIO:-16}"
FLEXIDEPTH_SCHEMA="${FLEXIDEPTH_SCHEMA:-auto}"

FLEX_ATTENTION_BLOCK_SIZE="${FLEX_ATTENTION_BLOCK_SIZE:-128}"
FLEX_ATTENTION_BACKEND="${FLEX_ATTENTION_BACKEND:-TRITON}"
FLEX_ATTENTION_COMPILE="${FLEX_ATTENTION_COMPILE:-True}"
FLEX_ATTENTION_COMPILE_MODE="${FLEX_ATTENTION_COMPILE_MODE:-max-autotune}"
FLEX_ATTENTION_MAX_MASK_CACHE_SIZE="${FLEX_ATTENTION_MAX_MASK_CACHE_SIZE:-256}"
FLEX_ATTENTION_SPARSE_PREFILL="${FLEX_ATTENTION_SPARSE_PREFILL:-False}"
FLEX_ATTENTION_SPARSE_DECODE="${FLEX_ATTENTION_SPARSE_DECODE:-True}"
FLEX_ATTENTION_DECODE_FULL_PREFIX_FASTPATH="${FLEX_ATTENTION_DECODE_FULL_PREFIX_FASTPATH:-True}"
FLEX_ATTENTION_DECODE_KV_ONLY_FASTPATH="${FLEX_ATTENTION_DECODE_KV_ONLY_FASTPATH:-True}"
FLEX_ATTENTION_DECODE_DENSE_FALLBACK="${FLEX_ATTENTION_DECODE_DENSE_FALLBACK:-True}"
FLEX_ATTENTION_DECODE_DENSE_ACTIVE_RATIO_THRESHOLD="${FLEX_ATTENTION_DECODE_DENSE_ACTIVE_RATIO_THRESHOLD:-0.5}"

FLEXIDEPTH_FOR_GENERATE="${FLEXIDEPTH_FOR_GENERATE:-True}"
FLEXIDEPTH_FOR_LOGLIKELIHOOD="${FLEXIDEPTH_FOR_LOGLIKELIHOOD:-False}"

ENABLE_ONE_STAGE_PREFILL="${ENABLE_ONE_STAGE_PREFILL:-True}"
PREFILL_KEEP_RATIO="${PREFILL_KEEP_RATIO:-0.25}"
PREFILL_MIN_KEEP="${PREFILL_MIN_KEEP:-16}"
PREFILL_CANDIDATE_LAYERS="${PREFILL_CANDIDATE_LAYERS:-15,16,17,18,19}"
PREFILL_MMR_LAMBDA="${PREFILL_MMR_LAMBDA:-0.5}"
PREFILL_MMR_SPATIAL_SIGMA="${PREFILL_MMR_SPATIAL_SIGMA:-0.02}"
PREFILL_MMR_COMBINE="${PREFILL_MMR_COMBINE:-add}"
PREFILL_IMPORTANCE_NORM_WEIGHT="${PREFILL_IMPORTANCE_NORM_WEIGHT:-1.0}"
PREFILL_IMPORTANCE_ANGLE_WEIGHT="${PREFILL_IMPORTANCE_ANGLE_WEIGHT:-0.0}"
PREFILL_IMPORTANCE_COMBINE="${PREFILL_IMPORTANCE_COMBINE:-add}"
PREFILL_REFORWARD="${PREFILL_REFORWARD:-False}"
PRUNE_PRE_KV="${PRUNE_PRE_KV:-False}"

PREFILL_CANDIDATE_LAYERS_ARG="${PREFILL_CANDIDATE_LAYERS//|/,}"
if [[ "${PREFILL_CANDIDATE_LAYERS_ARG}" != \[*\] ]]; then
  PREFILL_CANDIDATE_LAYERS_ARG="[${PREFILL_CANDIDATE_LAYERS_ARG}]"
fi

TASKS="${TASKS:-gqa,mme,mmbench_en_dev,pope,textvqa_val}"
OUTPUT_TASKS_SUFFIX="Tasks_$(echo "${TASKS}" | tr ',' '_')"
OUTPUT_TAG="${OUTPUT_TAG:-$(basename "${MODEL_PATH}")_${OUTPUT_TASKS_SUFFIX}}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/user/czx/open_source_proj/lmms-eval/outputs/logs/whole_test/llava/${OUTPUT_TAG}}"
LOG_PATH="${LOG_PATH:-${OUTPUT_PATH}/test.log}"
mkdir -p "${OUTPUT_PATH}"

MODEL_ARGS="pretrained=${MODEL_PATH},device_map=auto,attn_implementation=${ATTN_IMPLEMENTATION},flexidepth_runtime=v2,flexidepth_model_file=${FLEXIDEPTH_MODEL_FILE},flexidepth_layers=${FLEXIDEPTH_LAYERS},flexidepth_tau=${FLEXIDEPTH_TAU},flexidepth_router_ratio=${FLEXIDEPTH_ROUTER_RATIO},flexidepth_adapter_ratio=${FLEXIDEPTH_ADAPTER_RATIO},flexidepth_schema=${FLEXIDEPTH_SCHEMA},flex_attention_block_size=${FLEX_ATTENTION_BLOCK_SIZE},flex_attention_backend=${FLEX_ATTENTION_BACKEND},flex_attention_compile=${FLEX_ATTENTION_COMPILE},flex_attention_compile_mode=${FLEX_ATTENTION_COMPILE_MODE},flex_attention_max_mask_cache_size=${FLEX_ATTENTION_MAX_MASK_CACHE_SIZE},flex_attention_sparse_prefill=${FLEX_ATTENTION_SPARSE_PREFILL},flex_attention_sparse_decode=${FLEX_ATTENTION_SPARSE_DECODE},flex_attention_decode_full_prefix_fastpath=${FLEX_ATTENTION_DECODE_FULL_PREFIX_FASTPATH},flex_attention_decode_kv_only_fastpath=${FLEX_ATTENTION_DECODE_KV_ONLY_FASTPATH},flex_attention_decode_dense_fallback=${FLEX_ATTENTION_DECODE_DENSE_FALLBACK},flex_attention_decode_dense_active_ratio_threshold=${FLEX_ATTENTION_DECODE_DENSE_ACTIVE_RATIO_THRESHOLD},flexidepth_for_generate=${FLEXIDEPTH_FOR_GENERATE},flexidepth_for_loglikelihood=${FLEXIDEPTH_FOR_LOGLIKELIHOOD},enable_one_stage_prefill=${ENABLE_ONE_STAGE_PREFILL},prefill_keep_ratio=${PREFILL_KEEP_RATIO},prefill_min_keep=${PREFILL_MIN_KEEP},prefill_candidate_layers=${PREFILL_CANDIDATE_LAYERS_ARG},prefill_mmr_lambda=${PREFILL_MMR_LAMBDA},prefill_mmr_spatial_sigma=${PREFILL_MMR_SPATIAL_SIGMA},prefill_mmr_combine=${PREFILL_MMR_COMBINE},prefill_importance_norm_weight=${PREFILL_IMPORTANCE_NORM_WEIGHT},prefill_importance_angle_weight=${PREFILL_IMPORTANCE_ANGLE_WEIGHT},prefill_importance_combine=${PREFILL_IMPORTANCE_COMBINE},prefill_reforward=${PREFILL_REFORWARD},prune_pre_kv=${PRUNE_PRE_KV}"

echo "Saving outputs to ${OUTPUT_PATH}"
echo "Logging to ${LOG_PATH}"

accelerate launch --num_processes=1 \
  -m lmms_eval \
  --model llava_hf \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix reproduce \
  --output_path "${OUTPUT_PATH}" \
  > "${LOG_PATH}" 2>&1 &
