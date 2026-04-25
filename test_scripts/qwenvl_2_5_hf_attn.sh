# install lmms_eval without building dependencies
conda deactivate
source /workspace/open_source_proj/lmms-eval/.venv/bin/activate
cd /workspace/open_source_proj/lmms-eval;
# pip install --no-deps -U -e .
python -c "import transformers, inspect; print(transformers.__file__)"

export TRANSFORMERS_VERBOSITY=info
export TRANSFORMERS_USE_PYTORCH_OPERATORS=1

# 定义要测试的层
layers=(20 21 22 23 24 25 26 27)

# 并行启动4个进程，每次处理4个不同的层
for i in {0..7}; do
    start_idx=$((i * 4))
    
    # 并行启动4个进程
    for j in {0..3}; do
        layer_idx=$((start_idx + j))
        
        # 检查是否超出层数范围
        if [ $layer_idx -ge 32 ]; then
            break
        fi
        
        layer=${layers[$layer_idx]}
        output_path="/workspace/open_source_proj/lmms-eval/outputs/logs/qwenvl/fastv_0_delete_layer_score_${layer}/"
        
        echo "Starting process for scoring_layer=${layer} on GPU 0"
        
        accelerate launch --num_processes=1 \
            -m lmms_eval \
            --model qwen2_5_vl \
            --model_args pretrained="/workspace/model/Qwen/Qwen2.5-VL-7B-Instruct,device_map=auto,attn_implementation=eager,scoring_layer=${layer}" \
            --tasks gqa_lite \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "layer_${layer}" \
            --output_path "${output_path}" \
            > "/workspace/open_source_proj/lmms-eval/temp/qwenvl_layer_${layer}.log" 2>&1 &
    done
    
    # 等待当前4个进程完成
    wait
    echo "Completed batch $((i+1))/7 (layers $start_idx-$((start_idx+3)))"
done

echo "All merging layer tests completed!"