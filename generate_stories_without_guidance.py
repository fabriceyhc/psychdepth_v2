from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
import json
import gc
import os

l = [
    ("/data2/yihewang/psychdepth_v2/LLaMA-Factory/saves/qwen2.5-7b/lora/gsm8k_shuffled/checkpoint-%d", "qwen-shuffled")
]

print("Initializing CUDA...")
torch.cuda.init()
print("CUDA Initialized.")

premises = pd.read_csv("../llm-psych-depth/data/premises.csv") 
story_num = 10
for ind in l:
    # name_of_trial = ind.split('/')[0]
    # name_of_trial = "llama-shuffled"
    lora_dir, name_of_trial = ind
    # base_model = "/data2/yihewang/models/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"
    # lora_dir = f"/data2/yihewang/psychdepth_v2/LLaMA-Factory/saves/qwen2.5-7b/lora/{ind}"
    # lora_dir = "/data2/yihewang/psychdepth_v2/LLaMA-Factory/saves/llama3.1-8b/lora/gsm8k-shuffled/checkpoint-100"
    base_model = "/data2/.shared_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    # base_model = "../../.shared_models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/4281e96c7cf5ab6b312ef0cb78373efa3976a9dd"
    
    data = []
    
    for i in range(50, 650, 50):
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        lora_dir_n = lora_dir % i
        if ind != 'base':
            model = PeftModel.from_pretrained(model, lora_dir_n)
            model = model.merge_and_unload() 

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=600,
            temperature=1,
            do_sample=True
        )

        # 对话模板
        def format_prompt(message):
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {message}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        for index, row in premises.iterrows():
            print('index', index)
            premise = f"""You are a seasoned writer who has won several accolades for your emotionally rich stories. When you write, you delve deep into the human psyche, pulling from the reservoir of universal experiences that every reader, regardless of their background, can connect to. Your writing is renowned for painting vivid emotional landscapes, making readers not just observe but truly feel the world of your characters. Every piece you produce aims to draw readers in, encouraging them to reflect on their own lives and emotions. Your stories are a complex tapestry of relationships, emotions, and conflicts, each more intricate than the last. 

        Now write a 500-word story on the following prompt: 
        {row['premise']}
        Only respond with the story.
        """
            prompt = format_prompt(premise)
            for i in range(story_num):
                output = pipe(prompt)[0]['generated_text']
                response = output.split("<|eot_id|>")[-1].strip().split("<|end_header_id|>")[-1].strip()
                data.append({
                    'premise_id': index,
                    'premise': row['premise'],
                    'text': response,
                    'ckpt': lora_dir_n
                })
                print(i)
                print('already generated', len(data), 'stories')
                data_ = pd.DataFrame(data)
                data_['story_id'] = data_.index
                data_.to_csv(f"stories_{name_of_trial}_add.csv")
                break
            break
            
        del model
        del pipe
        torch.cuda.empty_cache()
        gc.collect()