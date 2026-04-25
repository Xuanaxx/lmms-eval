# install lmms_eval without building dependencies
source /workspace/open_source_proj/uv_env/.venv/bin/activate
cd /workspace/open_source_proj/lmms-eval;
# pip install --no-deps -U -e .
python -c "import transformers, inspect; print(transformers.__file__)"

export TRANSFORMERS_VERBOSITY=info
export TRANSFORMERS_USE_PYTORCH_OPERATORS=1  # 任意 env 以确保 handler 初始化
# Run and exactly reproduce llava_v1.5 results!
# mme as an example
# OUTPUT_FOLDER="officail_skip_src2_tgt6_scoring12_rss_beta_1.0_64tokens"
OUTPUT_FOLDER="officail_vanilla_llava_v1.5"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/workspace/model/liuhaotian/llava-v1.5-7b,device_map=auto,attn_implementation=eager"   \
    --tasks gqa,mme,mmbench,pope,textvqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix reproduce \
    --output_path "/workspace/open_source_proj/lmms-eval/outputs/logs/${OUTPUT_FOLDER}/" \
    > "/workspace/open_source_proj/lmms-eval/temp/${OUTPUT_FOLDER}.log" 2>&1

# gqa,mme,mmbench,pope,textvqa