#!bin/bash

set -x
source /workspace/open_source_proj/uv_env/.venv/bin/activate
# PAPER_TABLE=gqa,mmbench_en_dev,mme,pope,scienceqa_img,textvqa_val,seedbench,mmvet
# PAPER_TABLE=pope
cd /workspace/open_source_proj/lmms-eval;
# pip install --no-deps -U -e .
python -c "import transformers, inspect; print(transformers.__file__)"

export TRANSFORMERS_VERBOSITY=info
export TRANSFORMERS_USE_PYTORCH_OPERATORS=1  # 任意 env 以确保 handler 初始化
# Run and exactly reproduce llava_v1.5 results!
# mme as an example
PAPER_TABLE=gqa,mmbench,mme,pope,textvqa
LOG_DIR=./logs_final
RUN_NAME=SCOPE_LLaVA_7b_token_$1

ALPHA=1.0 BASELINE=SCOPE SUBSET_RATIO=$1 accelerate launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/workspace/model/liuhaotian/llava-v1.5-7b,device_map=auto"   \
    --tasks $PAPER_TABLE \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/$RUN_NAME
