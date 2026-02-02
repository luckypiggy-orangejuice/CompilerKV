#!/bin/bash

folders=(/DynamicKV/results/qwen2-7b-instruct_streamingllm_512_32_7_maxpool)

for folder in "${folders[@]}"; do
    python3 /DynamicKV/run/longbench/eval.py --results_dir "$folder" & 
done