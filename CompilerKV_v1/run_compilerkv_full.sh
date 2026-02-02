#!/bin/bash
# Run CompilerKV full evaluation on LongBench
# This script runs the full LongBench benchmark with CompilerKV

cd /root/autodl-tmp/compilerkv/Base

# Model and method configuration
MODEL_PATH="models/Mistral-7B-Instruct-v0.2"
MODEL_NAME="Mistral-7B-Instruct-v0.2"
METHOD="compilerkv"

# KV compression parameters
MAX_CAPACITY_PROMPT=512
WINDOW_SIZE=64
KERNEL_SIZE=7
POOLING="avgpool"
RADIO_MAX=10.0

# Paths
DATA_DIR="data/LongBench"
SAVE_DIR="results"

echo "=========================================="
echo "Starting CompilerKV Full LongBench Evaluation"
echo "Model: ${MODEL_NAME}"
echo "Method: ${METHOD}"
echo "Max capacity: ${MAX_CAPACITY_PROMPT}"
echo "=========================================="

# Run prediction
python run/longbench/pred.py \
    --model_path "${MODEL_PATH}" \
    --model_name "${MODEL_NAME}" \
    --method "${METHOD}" \
    --dataset_file "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --max_capacity_prompts ${MAX_CAPACITY_PROMPT} \
    --window_size ${WINDOW_SIZE} \
    --kernel_sizes ${KERNEL_SIZE} \
    --pooling "${POOLING}" \
    --radio_max ${RADIO_MAX} \
    --attn_implementation "flash_attention_2" \
    --eval_batch_size 1

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: ${SAVE_DIR}"
echo "=========================================="
