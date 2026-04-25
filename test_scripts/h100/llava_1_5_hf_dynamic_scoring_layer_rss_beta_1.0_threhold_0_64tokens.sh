# install lmms_eval without building dependencies
# ps aux | grep gqa | grep -v grep | awk '{print $2}' | xargs kill -9
conda deactivate ; source /home/user/czx/uv_env/.tokencompression/bin/activate
cd /home/user/czx/open_source_proj/lmms-eval;
# pip install --no-deps -U -e .
python -c "import transformers, inspect; print(transformers.__file__)"

export TRANSFORMERS_VERBOSITY=info
export TRANSFORMERS_USE_PYTORCH_OPERATORS=1
export OPENAI_API_KEY="sk-YP2RHFk6AQGUWtgCaMG9hdm685ZtkcFX1Uf5vplQjzI1VHCc"
export OPENAI_API_URL="https://xiaoai.plus/v1/chat/completions"
export CUDA_VISIBLE_DEVICES=7

# MODEL_PATH="/data1/czx/MLLM_Token_Compression_Workdir/output/llava_flexidepth_ft7_sum_type_linear"
# MODEL_PATH="/data1/czx/MLLM_Token_Compression_Workdir/output/llava_flexidepth_ft_origin_16_31_all"
# MODEL_PATH="/home/user/czx/model/llava-hf/llava-1.5-7b-hf"
# MODEL_PATH="/data1/czx/MLLM_Token_Compression_Workdir/output/llava_pre_scope_64_tokens_post_flexidepth_sum_type_linear" 
MODEL_PATH="/data1/czx/MLLM_Token_Compression_Workdir/output/llava_post_flexidepth_sum_type_linear_decode_optimized" 

# Run and exactly reproduce llava_v1.5 results!
TASKS="gqa,mme,mmbench_en_dev,pope,textvqa_val"
# TASKS="mmbench_en_dev,pope,textvqa_val"
# TASKS="gqa"

OUTPUT_TASKS_SUFFIX=Tasks_$(echo ${TASKS} | tr ',' '_')
# OUTPUT_FOLDER="dynamic_scoring_layer_rss_beta_1.0_threhold_0_64tokens"
# OUTPUT_FOLDER="dynamic_scoring_layer_mmr_add_lambda_0.5_spatial_sigma_0.02_64tokens_${OUTPUT_TASKS_SUFFIX}"
# OUTPUT_FOLDER="flexidepth_ft7_sum_type_square_norm_type_linear_ckpt20791_16_31_${OUTPUT_TASKS_SUFFIX}"
# OUTPUT_FOLDER="flexidepth_ft5_ckpt20791_0_6_16_31_visual_pruned_64tokenss_${OUTPUT_TASKS_SUFFIX}"
# OUTPUT_FOLDER="dynamic_scoring_layer_12_18_mmr_add_lambda_0.5_spatial_sigma_0.02_64tokens_oneforward_${OUTPUT_TASKS_SUFFIX}"
# OUTPUT_FOLDER="flexidepth_ft_origin_16_31_all_${OUTPUT_TASKS_SUFFIX}"
# OUTPUT_FOLDER="pre_scope_64_tokens_post_flexidepth_sum_type_linear_${OUTPUT_TASKS_SUFFIX}"
OUTPUT_FOLDER="post_flexidepth_sum_type_linear_decode_optimized_${OUTPUT_TASKS_SUFFIX}"

OUTPUT_PATH="/home/user/czx/open_source_proj/lmms-eval/outputs/logs/whole_test/llava/${OUTPUT_FOLDER}/"
LOG_PATH="/home/user/czx/open_source_proj/lmms-eval/outputs/logs/whole_test/llava/${OUTPUT_FOLDER}/test.log"
mkdir -p "${OUTPUT_PATH}"

# gqa,mme,mmbench_en_dev,pope,textvqa_val
echo "Saving outputs to ${OUTPUT_PATH}"
echo "Logging to ${LOG_PATH}"

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_hf \
    --model_args pretrained="${MODEL_PATH},device_map=auto,attn_implementation=eager" \
    --tasks ${TASKS} \
    --batch_size 1 \
    --attn_implementation flash_attention_2 \
    --log_samples \
    --log_samples_suffix reproduce \
    --output_path "${OUTPUT_PATH}" \
    > "${LOG_PATH}" 2>&1 &

# python3 -m lmms_eval \
#     --model llava_hf \
#     --model_args pretrained="${MODEL_PATH},device_map=auto,attn_implementation=eager" \
#     --tasks ${TASKS} \
#     --batch_size 1 \
#     --limit 10 \
#     --log_samples \
#     --log_samples_suffix reproduce \
#     --output_path "${OUTPUT_PATH}"