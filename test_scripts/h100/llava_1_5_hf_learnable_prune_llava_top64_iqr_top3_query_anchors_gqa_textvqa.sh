#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
    conda deactivate >/dev/null 2>&1 || true
fi
source /home/user/czx/uv_env/.tokencompression/bin/activate

REPO_ROOT="${REPO_ROOT:-/data2/chenzixuan/open_source_projects/lmms-eval}"
LLAVA_REPO="${LLAVA_REPO:-/data2/chenzixuan/open_source_projects/LLaVA_token_compression}"
export PYTHONPATH="${LLAVA_REPO}:${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
export TRANSFORMERS_USE_PYTORCH_OPERATORS="${TRANSFORMERS_USE_PYTORCH_OPERATORS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LEARNABLE_PRUNE_CHECKPOINT="${LEARNABLE_PRUNE_CHECKPOINT:-/data2/chenzixuan/train_output/llava_learnable_prune_budgeted_CE_KL_topk_hardening_top64_iqr_top3/checkpoint-1000}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-YP2RHFk6AQGUWtgCaMG9hdm685ZtkcFX1Uf5vplQjzI1VHCc}"
export OPENAI_API_URL="${OPENAI_API_URL:-https://xiaoai.plus/v1/chat/completions}"

MODEL_PATH="${MODEL_PATH:-/data2/chenzixuan/model/llava-hf/llava-1.5-7b-hf}"

TASKS="${TASKS:-gqa,vqav2,mmbench_en_dev,pope,textvqa_val,refcoco,refcoco+,refcocog,scienceqa}"
OUTPUT_TASKS_SUFFIX="Tasks_$(echo "${TASKS}" | tr ',' '_')"
OUTPUT_FOLDER="llava_learnable_prune_budgeted_topk_hardening_top64_iqr_top3_${OUTPUT_TASKS_SUFFIX}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/test_outputs/logs/whole_test/llava/${OUTPUT_FOLDER}/}"
LOG_PATH="${LOG_PATH:-${OUTPUT_PATH}/test.log}"
mkdir -p "${OUTPUT_PATH}"

MODEL_ARGS="pretrained=${MODEL_PATH},learnable_prune_model=true,device_map=auto,attn_implementation=sdpa"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

echo "Tasks: ${TASKS}"
echo "Model path: ${MODEL_PATH}"
echo "LLaVA repo: ${LLAVA_REPO}"
echo "Learnable-prune checkpoint: ${LEARNABLE_PRUNE_CHECKPOINT}"
echo "Model args: ${MODEL_ARGS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Saving outputs to ${OUTPUT_PATH}"
echo "Logging to ${LOG_PATH}"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    "${LIMIT_ARGS[@]}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "llava_learnable_prune_budgeted_topk_hardening_top64_iqr_top3_${OUTPUT_TASKS_SUFFIX}" \
    --output_path "${OUTPUT_PATH}" \
    > "${LOG_PATH}" 2>&1

if grep -E "Error during evaluation|Traceback|Error .* in generating" "${LOG_PATH}" >/dev/null; then
    echo "Evaluation log contains errors. See ${LOG_PATH}" >&2
    exit 1
fi

echo "Finished. See ${LOG_PATH} and ${OUTPUT_PATH}"
