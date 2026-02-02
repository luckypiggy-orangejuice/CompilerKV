
SAVE_DIR="/DynamicKV/results"
MODEL_PATH="/DynamicKV/models/qwen2-7b-instruct"
METHOD="DynamicKV_v11"
# MAX_CAPACITY_PROMPTS=1024
WINDOW_SIZE=8
KERNEL_SIZES=7
POOLING="maxpool"
LOG_FILE="/DynamicKV/logs/"
mcps=(64 96 128 256 512 1024 2048)

for mcp in "${mcps[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 /DynamicKV/run/longbench/pred.py \
        --save_dir $SAVE_DIR \
        --model_path $MODEL_PATH \
        --method $METHOD \
        --max_capacity_prompts $mcp \
        --window_size $WINDOW_SIZE \
        --kernel_sizes $KERNEL_SIZES \
        --pooling $POOLING \
        > ${LOG_FILE}"${METHOD}_${mcp}_${WINDOW_SIZE}_${KERNEL_SIZES}_${POOLING}_${MODEL_PATH##*/}" 2>&1
done