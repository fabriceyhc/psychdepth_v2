# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pipeline.run \
#   --base_model meta-llama/Llama-3.2-1B-Instruct \
#   --fine_tuned_model meta-llama/Llama-3.2-1B-Instruct \
#   --eval_dataset aime \
#   --datasets AIME_1983_2024_sft \

# CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
#   --base_model meta-llama/Llama-3.2-1B-Instruct \
#   --fine_tuned_model meta-llama/Llama-3.2-1B-Instruct \
#   --eval_dataset aime \
#   --datasets AIME_1983_2024_sft \
#   --strategy ./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PDSMultiScore_handpicked-demos=10_persona.json

CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --stage sft \
  --template llama3 \
  --eval_dataset aime \
  --datasets AIME_1983_2024_sft \
  --strategy ./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PDSMultiScore_handpicked-demos=10_persona.json
