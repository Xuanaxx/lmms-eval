#!/usr/bin/env bash
set -euo pipefail

# Re-run the failed avg-320 shard on one H100 with exactly one evaluation
# process.  The original run used PER_GPU_PROCESS_NUM=2 on GPU 6, so two
# complete LLaVA-NeXT processes competed for the same device and caused OOM.

BASE_TEST_SCRIPT=${BASE_TEST_SCRIPT:-"/data1/chenzixuan/MLLM_Token_Compression_Workdir/test_scripts/h100/llava_next_learnable_prune_lightweight_scope_finalwipe_query_anchors.sh"}
GPU=${GPU:-0}
TASKS="pope,textvqa_val,mme,scienceqa"
MIN_FREE_GPU_MIB=${MIN_FREE_GPU_MIB:-45000}

ORIGINAL_OUTPUT_ROOT="/data1/chenzixuan/open_source_projects/lmms-eval/test_outputs/logs/whole_test/llava/llava_next_learnable_prune_lightweight_scope_finalwipe_top160_layer18_query_anchors_enable_predictor_1_Tasks_gqa_vqav2_val_mmbench_en_dev_mmbench_cn_dev_pope_textvqa_val_mme_scienceqa_avg320_l0_680_l12_160_l25"
OUTPUT_PATH=${OUTPUT_PATH:-"${ORIGINAL_OUTPUT_ROOT}/rerun_single_gpu${GPU}_pope_textvqa_val_mme_scienceqa"}
LOG_PATH=${LOG_PATH:-"${OUTPUT_PATH}/test.log"}

[[ -f "${BASE_TEST_SCRIPT}" ]] || {
    echo "Base LLaVA-NeXT test script not found: ${BASE_TEST_SCRIPT}" >&2
    exit 1
}

mkdir -p "${OUTPUT_PATH}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "nvidia-smi is unavailable; run this script on a GPU node." >&2
        exit 1
    }
    FREE_GPU_MIB=$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ "${FREE_GPU_MIB}" =~ ^[0-9]+$ ]] || {
        echo "Could not query free memory for physical GPU ${GPU}." >&2
        exit 1
    }
    if (( FREE_GPU_MIB < MIN_FREE_GPU_MIB )); then
        echo "GPU ${GPU} has only ${FREE_GPU_MIB} MiB free; at least ${MIN_FREE_GPU_MIB} MiB is required." >&2
        echo "Choose an idle H100 with GPU=<index>, or stop competing processes before retrying." >&2
        exit 1
    fi
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PER_GPU_PROCESS_NUM=0
export AVG_TOKEN_BUDGET=320
export TASKS
export OUTPUT_PATH
export LOG_PATH
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"expandable_segments:True"}

echo "Re-running failed shard with one process on physical GPU ${GPU}"
echo "Tasks: ${TASKS}"
echo "Output: ${OUTPUT_PATH}"
echo "Required free GPU memory: ${MIN_FREE_GPU_MIB} MiB"

exec bash "${BASE_TEST_SCRIPT}"
