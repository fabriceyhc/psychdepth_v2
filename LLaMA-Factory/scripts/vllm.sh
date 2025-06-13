export CUDA_VISIBLE_DEVICES=0
FACT=/data2/yihewang/psychdepth_v2/LLaMA-Factory
# MODEL=Qwen/Qwen2.5-7B-Instruct
MODEL=meta-llama/Llama-3.1-8B-Instruct

for ((i=200; i<=300; i+=50)); 
do
  python scripts/vllm_infer.py \
    --model_name_or_path $MODEL \
    --adapter_name_or_path $FACT/saves/llama3-8b/lora/sft/shuf-0507/checkpoint-1250 \
    --temperature 0 \
    --dataset gsm8k_test \
    --save_name /data2/yihewang/psychdepth_v2/LLaMA-Factory/vllm/new_test/temp0_qwen_gsm8k_answer_$i.jsonl \
    --max_new_tokens 10
done
