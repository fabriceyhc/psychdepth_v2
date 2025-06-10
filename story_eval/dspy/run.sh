# CUDA_VISIBLE_DEVICES=0 python -m story_eval.dspy.singlescore.optimize --model_id meta-llama/Llama-3.1-8B-Instruct
# CUDA_VISIBLE_DEVICES=0 python -m story_eval.dspy.multiscore.optimize  --model_id meta-llama/Llama-3.1-8B-Instruct

# CUDA_VISIBLE_DEVICES=0 python -m story_eval.dspy.singlescore.optimize --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# CUDA_VISIBLE_DEVICES=4,5 python -m story_eval.dspy.multiscore.optimize  --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m story_eval.dspy.singlescore.optimize --model_id meta-llama/Llama-3.1-70B-Instruct
# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m story_eval.dspy.multiscore.optimize  --model_id meta-llama/Llama-3.1-70B-Instruct

# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m story_eval.dspy.singlescore.optimize --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# CUDA_VISIBLE_DEVICES=4,5 python -m story_eval.dspy.multiscore.optimize  --model_id deepseek-ai/DeepSeek-R1-Distill-Llama-70B


# CUDA_VISIBLE_DEVICES=0,3,5,6 python -m story_eval.dspy.multiscore.annotate --strategy ./story_eval/dspy/multiscore/optimized_prompts/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/MIPROv2_Predict-PsychDepthAssessment_kmean-demos=7_persona.json
# CUDA_VISIBLE_DEVICES=0,3,5,6 python -m story_eval.dspy.multiscore.annotate --strategy ./story_eval/dspy/multiscore/optimized_prompts/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/MIPROv2_ChainOfThought-PsychDepthAssessment_handpicked-demos=7_persona.json
# CUDA_VISIBLE_DEVICES=0,3,5,6 python -m story_eval.dspy.multiscore.annotate --strategy ./story_eval/dspy/multiscore/optimized_prompts/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/MIPROv2_Predict-PsychDepthAssessment_handpicked-demos=7_persona.json
# CUDA_VISIBLE_DEVICES=0,3,5,6 python -m story_eval.dspy.multiscore.annotate --strategy ./story_eval/dspy/multiscore/optimized_prompts/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/MIPROv2_ChainOfThought-PsychDepthAssessment_handpicked-demos=10_persona.json
# CUDA_VISIBLE_DEVICES=0,3,5,6 python -m story_eval.dspy.multiscore.annotate --strategy ./story_eval/dspy/multiscore/optimized_prompts/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/MIPROv2_Predict-PsychDepthAssessment_handpicked-demos=10_persona.json
# CUDA_VISIBLE_DEVICES=1 python -m story_eval.dspy.multiscore.annotate --strategy ./story_eval/dspy/multiscore/optimized_prompts/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/MIPROv2_Predict-PsychDepthAssessment_handpicked-demos=10.json --dataset stories.csv --output evaluation.csv
# CUDA_VISIBLE_DEVICES=3 python -m story_eval.dspy.multiscore.annotate --strategy ./story_eval/dspy/multiscore/optimized_prompts/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/MIPROv2_Predict-PsychDepthAssessment_handpicked-demos=5_persona.json

CUDA_VISIBLE_DEVICES=1,2 python -m story_eval.dspy.multiscore.annotate_participant --output ./story_eval/dspy/dspy_annotations/llama-3.1-8B_persona/
# CUDA_VISIBLE_DEVICES=0,2,3,5 python -m story_eval.dspy.multiscore.annotate --strategy ./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-70B-Instruct/MIPROv2_Predict-PsychDepthAssessment_handpicked-demos=10_persona.json
# CUDA_VISIBLE_DEVICES=0,2,3,5 python -m story_eval.dspy.multiscore.annotate --strategy ./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-70B-Instruct/MIPROv2_Predict-PsychDepthAssessment_handpicked-demos=10_persona.json