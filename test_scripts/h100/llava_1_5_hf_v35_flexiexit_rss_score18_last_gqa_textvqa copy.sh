#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
    conda deactivate || true
fi
source /home/user/czx/uv_env/.tokencompression/bin/activate

REPO_ROOT="/data2/chenzixuan/open_source_projects/lmms-eval"
cd "${REPO_ROOT}"

export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
export TRANSFORMERS_USE_PYTORCH_OPERATORS="${TRANSFORMERS_USE_PYTORCH_OPERATORS:-1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-YP2RHFk6AQGUWtgCaMG9hdm685ZtkcFX1Uf5vplQjzI1VHCc}"
export OPENAI_API_URL="${OPENAI_API_URL:-https://xiaoai.plus/v1/chat/completions}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL_PATH="${MODEL_PATH:-/data2/chenzixuan/train_output/llava_v35_prefill_flexiexit_router_adapter_decode_layers_20-31_router_16_adapter_16_alpha_0.001_norm_linear_token_constant}"
CUSTOM_MODEL_FILE="${CUSTOM_MODEL_FILE:-/data2/chenzixuan/MLLM_Token_Compression_Workdir/src/LLaVA_flexidepth/final_prefill_decode/modeling_llava_v35_flexiexit_hybrid.py}"

TASKS="${TASKS:-gqa,mme,mmbench_en_dev,pope,textvqa_val}"
# TASKS="${TASKS:-textvqa_val}"

OUTPUT_TASKS_SUFFIX="Tasks_$(echo "${TASKS}" | tr ',' '_')"
OUTPUT_FOLDER="v35_flexiexit_three_stage_idx0_quadtree_on_idx6_recover_rss_topk_idx18_rss_score_${OUTPUT_TASKS_SUFFIX}"
OUTPUT_PATH="${OUTPUT_PATH:-/data2/chenzixuan/open_source_projects/lmms-eval/test_outputs/logs/whole_test/llava/${OUTPUT_FOLDER}/}"
LOG_PATH="${LOG_PATH:-${OUTPUT_PATH}/test.log}"
mkdir -p "${OUTPUT_PATH}"

MODEL_ARGS="pretrained=${MODEL_PATH},custom_model_file=${CUSTOM_MODEL_FILE},device_map=auto,attn_implementation=eager,stage1_merge_layer_idx=0,recover_layer_idx=6,final_prune_layer_idx=18,stage1_target_count=288,recover_topk_target_count=288,visual_token_target_count=32,scoring_layer_idx=18,attn_anchor=last"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

echo "Tasks: ${TASKS}"
echo "Model path: ${MODEL_PATH}"
echo "Custom model file: ${CUSTOM_MODEL_FILE}"
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
    --log_samples_suffix "v35_flexiexit_three_stage_idx0_quadtree_on_idx6_recover_rss_topk_idx18_rss_score_${OUTPUT_TASKS_SUFFIX}" \
    --output_path "${OUTPUT_PATH}" \
    > "${LOG_PATH}" 2>&1

if grep -E "Error during evaluation|Traceback|Error .* in generating" "${LOG_PATH}" >/dev/null; then
    echo "Evaluation log contains errors. See ${LOG_PATH}" >&2
    exit 1
fi

echo "Finished. See ${LOG_PATH} and ${OUTPUT_PATH}"
