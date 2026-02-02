
SAVE_DIR="/DynamicKV/results"
MODEL_PATH="/DynamicKV/models/mistral-7b-instruct-v0.2"
METHOD="DynamicKV_v11"
# MAX_CAPACITY_PROMPTS=128
WINDOW_SIZE=8
KERNEL_SIZES=7
POOLING="maxpool"
RADIO_MAX=10
LOG_FILE="/DynamicKV/logs/"
mcps=(64 96 128 256 512 1024 2048 4096)


for mcp in "${mcps[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 /DynamicKV/run/longbench/pred.py \
        --save_dir $SAVE_DIR \
        --model_path $MODEL_PATH \
        --method $METHOD \
        --max_capacity_prompts $mcp \
        --window_size $WINDOW_SIZE \
        --kernel_sizes $KERNEL_SIZES \
        --pooling $POOLING \
        --radio_max $RADIO_MAX \
        > ${LOG_FILE}"${METHOD}_${mcp}_${RADIO_MAX}_${WINDOW_SIZE}_${KERNEL_SIZES}_${POOLING}_${MODEL_PATH##*/}" 2>&1
done