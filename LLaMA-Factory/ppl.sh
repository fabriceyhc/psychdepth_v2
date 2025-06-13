#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export UNSLOTH_RETURN_LOGITS=1

for i in $(seq 50 50 300); do

    echo "Evaluating checkpoint-$i..."
    python scr*/stat*/cal_ppl.py \
        --model_name_or_path /data2/.shared_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
        --adapter_name_or_path /data2/yihewang/psychdepth_v2/LLaMA-Factory/saves/qwen2.5-7b/lora/gsm8k_answer/checkpoint-$i \
        --save_name /data2/yihewang/psychdepth_v2/LLaMA-Factory/ppl/qwen_answer_${i}.json \
        --dataset deepseek_stories
# done

    # echo "Evaluating checkpoint-$i..."
    # python scr*/stat*/cal_ppl.py \
    #     --model_name_or_path /data2/yihewang/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/model \
    #     --adapter_name_or_path /data2/yihewang/psychdepth_v2/LLaMA-Factory/saves/llama3.1-8b/lora/gsm8k_shuffled/checkpoint-$i \
    #     --save_name /data2/yihewang/psychdepth_v2/LLaMA-Factory/ppl/llama_shuffled_${i}.json \
    #     --dataset deepseek_stories
done
