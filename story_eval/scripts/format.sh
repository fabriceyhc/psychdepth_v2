python -m story_eval.format \
    --input_best_llm ./story_eval/dspy/dspy_annotations/meta-llama_Llama-3.1-70B-Instruct_predictions_MIPROv2_Predict-PsychDepthAssessment_handpicked-demos=7_persona.csv \
    --input_worst_llm ./story_eval/dspy/dspy_annotations/deepseek-ai_DeepSeek-R1-Distill-Llama-8B_predictions_MIPROv2_Predict-PsychDepthAssessment_demos=10.csv \
    --output ./data