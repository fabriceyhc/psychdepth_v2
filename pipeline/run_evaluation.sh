CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pipeline.run \
  --base_model Qwen/Qwen2.5-Math-7B-Instruct \
  --fine_tuned_model open-r1/OpenR1-Qwen-7B \
  --eval_dataset aime 

# CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
#   --base_model meta-llama/Llama-3.2-1B-Instruct \
#   --fine_tuned_model meta-llama/Llama-3.2-1B-Instruct \
#   --eval_dataset aime \
#   --datasets AIME_1983_2024_sft \
#   --strategy ./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PDSMultiScore_handpicked-demos=10_persona.json

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pipeline.run \
#   --base_model meta-llama/Llama-3.2-1B-Instruct \
#   --stage sft {sft, dpo, kto} \
#   --template llama3 \
#   --eval_dataset aime {evaluation math dataset, currently include "aime" and "math500"} \
#   --datasets AIME_1983_2024_sft {Post training dataset} \

# CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
#   --base_model meta-llama/Llama-3.2-1B-Instruct \
#   --stage sft \
#   --template llama3 \
#   --eval_dataset aime \
#   --datasets AIME_1983_2024_sft \
#   --strategy ./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PDSMultiScore_handpicked-demos=10_persona.json
