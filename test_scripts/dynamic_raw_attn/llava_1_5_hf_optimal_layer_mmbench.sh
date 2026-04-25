# install lmms_eval without building dependencies
conda deactivate
source /home/user/czx/uv_env/.tokencompression/bin/activate
cd /home/user/czx/open_source_proj/lmms-eval;
# pip install --no-deps -U -e .
python -c "import transformers, inspect; print(transformers.__file__)"

export TRANSFORMERS_VERBOSITY=info
export TRANSFORMERS_USE_PYTORCH_OPERATORS=1
export CUDA_VISIBLE_DEVICES=6
export OPENAI_API_KEY="sk-YP2RHFk6AQGUWtgCaMG9hdm685ZtkcFX1Uf5vplQjzI1VHCc"
export OPENAI_API_URL="https://xiaoai.plus/v1/chat/completions"

LIMIT_DATA_NUM=500
PROCESS_NUM=4 # 新增超参数：决定并行的进程数量
TASK="mmbench_en_dev"
LOG_DIR="/home/user/czx/open_source_proj/lmms-eval/tmp/${TASK}/"
mkdir -p ${LOG_DIR}
# 定义要测试的层
layers=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
# layers=(12 13 14 15)
# layers=(19 24 26 30 31)
len_layers=${#layers[@]}
loops=$(( (len_layers + PROCESS_NUM - 1) / PROCESS_NUM )) # 动态计算循环次数 (向上取整)
# 并行启动多个进程
for i in $(seq 0 $((loops-1))); do
    start_idx=$((i * PROCESS_NUM))
    
    # 并行启动进程
    for j in $(seq 0 $((PROCESS_NUM - 1))); do
        layer_idx=$((start_idx + j))
        
        # 检查是否超出层数数组范围
        if [ $layer_idx -ge $len_layers ]; then
            break
        fi
        
        layer=${layers[$layer_idx]}
        
        # 新建输出目录
        
        output_path="/home/user/czx/open_source_proj/lmms-eval/outputs/logs/llava_raw_attn_dynamic_${TASK}/scoring_layer_${layer}/"
        log_path="${LOG_DIR}/llava_raw_attn_layer_${layer}.log"
        mkdir -p ${output_path}
        METRICS_SAVE_PATH="${output_path}/metrics_layer_${layer}.json"

        echo "Starting process for scoring_layer=${layer} on GPU $CUDA_VISIBLE_DEVICES, logging to ${log_path}"
        
        accelerate launch --num_processes=1 \
            -m lmms_eval \
            --model llava_hf \
            --model_args pretrained="/home/user/czx/model/llava-hf/llava-1.5-7b-hf,device_map=auto,attn_implementation=eager,scoring_layer_idx=${layer},limit_data_num=${LIMIT_DATA_NUM},metrics_save_path=${METRICS_SAVE_PATH}" \
            --tasks ${TASK} \
            --limit ${LIMIT_DATA_NUM} \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "layer_${layer}" \
            --output_path "${output_path}" \
            > "${log_path}" 2>&1 &
    done
    
    # 等待当前批次进程完成
    wait
    echo "Completed batch $((i+1))/${loops} (layers $start_idx-$((start_idx+PROCESS_NUM-1)))"
done

echo "All merging layer tests completed for task ${TASK}!"