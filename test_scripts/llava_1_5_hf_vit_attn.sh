# install lmms_eval without building dependencies
conda deactivate
source /workspace/open_source_proj/lmms-eval/.venv/bin/activate
cd /workspace/open_source_proj/lmms-eval;
# pip install --no-deps -U -e .
python -c "import transformers, inspect; print(transformers.__file__)"

export TRANSFORMERS_VERBOSITY=info
export TRANSFORMERS_USE_PYTORCH_OPERATORS=1  # 任意 env 以确保 handler 初始化
# Run and exactly reproduce llava_v1.5 results!
# mme as an example
accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_hf \
    --model_args pretrained="/workspace/model/llava-hf/llava-1.5-7b-hf,device_map=auto,attn_implementation=eager"   \
    --tasks gqa_lite \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix reproduce \
    --output_path /workspace/open_source_proj/lmms-eval/outputs/logs/vit_attn/