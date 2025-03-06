# CUDA_VISIBLE_DEVICES=0 python -m story_eval.dspy.singlescore.optimize --model_id meta-llama/Llama-3.1-8B-Instruct
# CUDA_VISIBLE_DEVICES=0 python -m story_eval.dspy.multiscore.optimize  --model_id meta-llama/Llama-3.1-8B-Instruct

# CUDA_VISIBLE_DEVICES=0 python -m story_eval.dspy.singlescore.optimize --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B
CUDA_VISIBLE_DEVICES=4,5 python -m story_eval.dspy.multiscore.optimize  --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m story_eval.dspy.singlescore.optimize --model_id meta-llama/Llama-3.1-70B-Instruct
# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m story_eval.dspy.multiscore.optimize  --model_id meta-llama/Llama-3.1-70B-Instruct

# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m story_eval.dspy.singlescore.optimize --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# CUDA_VISIBLE_DEVICES=4,5 python -m story_eval.dspy.multiscore.optimize  --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-70B