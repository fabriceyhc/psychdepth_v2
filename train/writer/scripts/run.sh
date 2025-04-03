CUDA_VISIBLE_DEVICES=0,1 python -m train.writer.run \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --template llama3 \
  --stage sft \
  --datasets pdsv2_multiscore_sft 

CUDA_VISIBLE_DEVICES=0,1 python -m train.writer.run \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --template llama3 \
  --stage dpo \
  --datasets pdsv2_multiscore_dpo 

CUDA_VISIBLE_DEVICES=0,1 python -m train.writer.run \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --template llama3 \
  --stage kto \
  --datasets pdsv2_multiscore_kto 