#!/bin/bash
# Run CompilerKV evaluation on LongBench

# Model and method configuration
MODEL_PATH="models/Llama-3-8B-Instruct"
MODEL_NAME="llama-3-8b-instruct"
METHOD="compilerkv"  # Use our Stage1-Stage2-Stage3 method

# KV compression parameters
MAX_CAPACITY_PROMPT=2048  # Total KV budget
WINDOW_SIZE=64           # Observation window size (w_obs)
KERNEL_SIZE=7            # Smoothing kernel size
POOLING="avgpool"        # Pooling method
RADIO_MAX=10.0           # Max budget ratio

# Paths
DATA_DIR="../../data/LongBench"
SAVE_DIR="../../results/compilerkv"

# Datasets to evaluate
DATASETS=("narrativeqa" "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "musique" \
          "gov_report" "qmsum" "multi_news" "trec" "triviaqa" "samsum" \
          "passage_count" "passage_retrieval_en" "lcc" "repobench-p")

# Run evaluation
for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Evaluating on dataset: ${dataset}"
    echo "=========================================="
    
    python ../pred.py \
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
done

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to: ${SAVE_DIR}"
echo "=========================================="
