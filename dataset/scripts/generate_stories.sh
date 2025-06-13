# CUDA_VISIBLE_DEVICES=0 python -m dataset.generate_stories \
#   --backend_type transformers \
#   --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

CUDA_VISIBLE_DEVICES=0 python -m dataset.generate_stories \
  --backend_type transformers \
  --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# CUDA_VISIBLE_DEVICES=0 python -m dataset.generate_stories \
#   --backend_type llamacpp \
#   --llamacpp_model_path /data2/.shared_models/llama.cpp_models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf \
#   --llamacpp_n_ctx 8192

# CUDA_VISIBLE_DEVICES=2 python -m dataset.generate_stories \
#   --backend_type llamacpp \
#   --llamacpp_model_path /data2/.shared_models/llama.cpp_models/Llama-3.1-8B-Instruct-Q8_0.gguf \
#   --llamacpp_n_ctx 8192


 