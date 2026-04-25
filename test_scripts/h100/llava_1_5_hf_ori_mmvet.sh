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
export OPENAI_API_URL="${OPENAI_API_URL:-https://xiaoai.plus/v1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

MODEL_PATH="${MODEL_PATH:-/home/user/czx/model/llava-hf/llava-1.5-7b-hf}"

# TASKS="${TASKS:-gqa,mme,mmbench_en_dev,pope,textvqa_val}"
TASKS="${TASKS:-mmvet}"

OUTPUT_TASKS_SUFFIX="Tasks_$(echo "${TASKS}" | tr ',' '_')"
OUTPUT_FOLDER="ori_${OUTPUT_TASKS_SUFFIX}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/user/czx/open_source_proj/lmms-eval/outputs/logs/whole_test/llava/${OUTPUT_FOLDER}/}"
LOG_PATH="${LOG_PATH:-${OUTPUT_PATH}/test.log}"
mkdir -p "${OUTPUT_PATH}"

MODEL_ARGS="pretrained=${MODEL_PATH},device_map=auto,attn_implementation=eager"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

echo "Tasks: ${TASKS}"
echo "Model path: ${MODEL_PATH}"
echo "Model args: ${MODEL_ARGS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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
    --log_samples_suffix "ori_${OUTPUT_TASKS_SUFFIX}" \
    --output_path "${OUTPUT_PATH}" \
    > "${LOG_PATH}" 2>&1

if grep -E "Error during evaluation|Traceback|Error .* in generating" "${LOG_PATH}" >/dev/null; then
    echo "Evaluation log contains errors. See ${LOG_PATH}" >&2
    exit 1
fi

echo "Finished. See ${LOG_PATH} and ${OUTPUT_PATH}"
