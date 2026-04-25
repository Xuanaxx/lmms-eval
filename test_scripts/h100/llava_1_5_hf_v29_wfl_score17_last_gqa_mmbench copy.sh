#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
    conda deactivate || true
fi
source /home/user/czx/uv_env/.tokencompression/bin/activate

REPO_ROOT="/home/user/czx/open_source_proj/lmms-eval"

export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
export TRANSFORMERS_USE_PYTORCH_OPERATORS="${TRANSFORMERS_USE_PYTORCH_OPERATORS:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-YP2RHFk6AQGUWtgCaMG9hdm685ZtkcFX1Uf5vplQjzI1VHCc}"
export OPENAI_API_URL="${OPENAI_API_URL:-https://xiaoai.plus/v1/chat/completions}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL_PATH="${MODEL_PATH:-/home/user/czx/model/llava-hf/llava-1.5-7b-hf}"
TASKS="${TASKS:-gqa,mmbench_en_dev}"
SCORING_LAYER_IDX="${SCORING_LAYER_IDX:-18}"
VISUAL_TOKEN_KEEP_RATIO="${VISUAL_TOKEN_KEEP_RATIO:-0.112}"
PRUNING_MODE="${PRUNING_MODE:-wfl}"
ATTN_ANCHOR="${ATTN_ANCHOR:-last}"
POOL_TYPE="${POOL_TYPE:-avg}"
WFL_LAMBDA="${WFL_LAMBDA:-0.5}"
WFL_SIGMA="${WFL_SIGMA:-0.1}"
WFL_COVERAGE_WEIGHT_TYPE="${WFL_COVERAGE_WEIGHT_TYPE:-importance}"

OUTPUT_TASKS_SUFFIX="Tasks_$(echo "${TASKS}" | tr ',' '_')"
OUTPUT_FOLDER="v29_${PRUNING_MODE}_score${SCORING_LAYER_IDX}_${ATTN_ANCHOR}_keep${VISUAL_TOKEN_KEEP_RATIO}_${OUTPUT_TASKS_SUFFIX}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/user/czx/open_source_proj/lmms-eval/outputs/logs/whole_test/llava/${OUTPUT_FOLDER}/}"
LOG_PATH="${LOG_PATH:-${OUTPUT_PATH}/test.log}"
mkdir -p "${OUTPUT_PATH}"

MODEL_ARGS="pretrained=${MODEL_PATH},device_map=auto,attn_implementation=eager,scoring_layer_idx=${SCORING_LAYER_IDX},force_fixed_scoring_layer=true,attn_anchor=${ATTN_ANCHOR},pool_type=${POOL_TYPE},pruning_mode=${PRUNING_MODE},visual_token_keep_ratio=${VISUAL_TOKEN_KEEP_RATIO},visual_token_min_keep=16,wfl_lambda=${WFL_LAMBDA},wfl_sigma=${WFL_SIGMA},wfl_coverage_weight_type=${WFL_COVERAGE_WEIGHT_TYPE}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

echo "Tasks: ${TASKS}"
echo "Model path: ${MODEL_PATH}"
echo "Model args: ${MODEL_ARGS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "Saving outputs to ${OUTPUT_PATH}"
echo "Logging to ${LOG_PATH}"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_hf \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    "${LIMIT_ARGS[@]}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "v29_wfl_score${SCORING_LAYER_IDX}_${ATTN_ANCHOR}" \
    --output_path "${OUTPUT_PATH}" \
    > "${LOG_PATH}" 2>&1

if grep -E "Error during evaluation|Traceback|Error .* in generating" "${LOG_PATH}" >/dev/null; then
    echo "Evaluation log contains errors. See ${LOG_PATH}" >&2
    exit 1
fi

echo "Finished. See ${LOG_PATH} and ${OUTPUT_PATH}"
