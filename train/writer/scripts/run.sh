CUDA_VISIBLE_DEVICES=0,1 python -m train.writer.run \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --template llama3 \
  --stage sft \
  --datasets AIME_1983_2024_sft \
  --train_output_dir /data2/ruichenzheng/llamafactory_train/AIME_1983_2024_sft/meta-llama/Llama-3.2-1B-Instruct \
  --export_output_dir /data2/ruichenzheng/llamafactory_outputs/AIME_1983_2024_sft/meta-llama/Llama-3.2-1B-Instruct \
  --num_train_epochs 5
 

# CUDA_VISIBLE_DEVICES=0,1 python -m train.writer.run \
#   --model_name meta-llama/Llama-3.2-1B-Instruct \
#   --template llama3 \
#   --stage dpo \
#   --datasets pdsv2_multiscore_dpo 

# CUDA_VISIBLE_DEVICES=0,1 python -m train.writer.run \
#   --model_name meta-llama/Llama-3.2-1B-Instruct \
#   --template llama3 \
#   --stage kto \
#   --datasets pdsv2_multiscore_kto 