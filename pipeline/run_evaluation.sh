# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pipeline.run \
#   --base_model Qwen/Qwen2.5-7B-Instruct \
#   --fine_tuned_model InfiniAILab/OpenR1-Qwen-7B-SFT-Instruct \
#   --eval_dataset aime 

# CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
#   --base_model meta-llama/Llama-3.2-1B-Instruct \
#   --fine_tuned_model meta-llama/Llama-3.2-1B-Instruct \
#   --eval_dataset aime \
#   --datasets AIME_1983_2024_sft \
#   --strategy ./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PDSMultiScore_handpicked-demos=10_persona.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pipeline.run \
  --base_model microsoft/phi-4 \
  --fine_tuned_model microsoft/Phi-4-reasoning \
  --backend_type transformers \
  --eval_dataset math500 \
  --temparature 0.8 \
  --num_versions 3 \
  --base_math_eval false \
  --base_story_gen false \
  --base_story_eval true \
  --fine_tuned_math_eval false \
  --fine_tuned_story_gen false \
  --fine_tuned_story_eval true \

# CUDA_VISIBLE_DEVICES=1,2,5,6 python -m pipeline.run \
#   --base_model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
#   --fine_tuned_model reedmayhew/Llama-3.1-8B-claude-3.7-sonnet-reasoning-distilled \
#   --backend_type llama.cpp \
#   --eval_dataset math500 

# CUDA_VISIBLE_DEVICES=1,2,4,5 python -m pipeline.run \
#   --base_model Qwen/Qwen2.5-32B-Instruct \
#   --fine_tuned_model OpenPipe/Deductive-Reasoning-Qwen-32B \
#   --backend_type transformers \
#   --eval_dataset math500 

  # CUDA_VISIBLE_DEVICES=1,2,4,5 python -m pipeline.run \
  # --base_model microsoft/Phi-4-mini-instruct \
  # --fine_tuned_model microsoft/Phi-4-mini-reasoning \
  # --backend_type transformers \
  # --eval_dataset math500 

  # CUDA_VISIBLE_DEVICES=0,1 python -m pipeline.run \
  # --base_model microsoft/phi-4 \
  # --fine_tuned_model microsoft/Phi-4-reasoning \
  # --backend_type transformers \
  # --eval_dataset math500 
  
  # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pipeline.run \
  # --base_model Qwen/Qwen2.5-7B-Instruct  \
  # --fine_tuned_model ZMC2019/Qwen7B-MP \
  # --backend_type transformers \
  # --eval_dataset math500 

  # CUDA_VISIBLE_DEVICES=0,1 python -m pipeline.run \
  # --base_model microsoft/phi-4 \
  # --fine_tuned_model microsoft/Phi-4-reasoning \
  # --backend_type transformers \
  # --eval_dataset math500 

  # CUDA_VISIBLE_DEVICES=0,1 python -m pipeline.run \
  # --base_model meta-llama/Llama-3.1-8B-Instruct \
  # --fine_tuned_model EpistemeAI/Reasoning-Llama-3.1-CoT-RE1-NMT \
  # --backend_type transformers \
  # --eval_dataset math500 

  # TODO: add nemotron llama 8b with system prompt: detailed thinking off

  # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pipeline.run \
  # --base_model Qwen/Qwen2.5-32B-Instruct \
  # --fine_tuned_model nvidia/OpenMath-Nemotron-32B \
  # --backend_type transformers \
  # --eval_dataset math500 
  
# CUDA_VISIBLE_DEVICES=1,2,5,6 python -m pipeline.run \
#   --base_model Qwen/Qwen2.5-7B-Instruct \
#   --stage sft \
#   --template llama3 \
#   --eval_dataset math500 \
#   --datasets amc_aime_sft_ai_cot_solution \
#   --num_train_epochs 7

# CUDA_VISIBLE_DEVICES=0 python -m pipeline.run \
#   --base_model meta-llama/Llama-3.2-1B-Instruct \
#   --stage sft \
#   --template llama3 \
#   --eval_dataset aime \
#   --datasets AIME_1983_2024_sft \
#   --strategy ./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PDSMultiScore_handpicked-demos=10_persona.json
