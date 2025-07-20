from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re

# Load once at module import
MODEL_NAME = "codellama/CodeLlama-7b-hf"          # ← pick any causal‑LM you have
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)

def postprocess_completion(prompt: str, full_text: str) -> str:
    """
    Strip the echoed prompt and keep only the newly generated code body.
    """
    # Remove the prompt if the model repeated it
    if full_text.startswith(prompt):
        body = full_text[len(prompt):]
    else:
        # Fallback: keep everything after the first newline
        body = full_text.split("\n", 1)[-1]
    # Optional: cut off after two consecutive blank lines or '```'
    body = re.split(r"\n\s*\n|```", body)[0]
    # Ensure the body ends with a newline for cleaner concatenation
    if not body.endswith("\n"):
        body += "\n"
    return body

def generate_one_completion(prompt: str,
                            max_new_tokens: int = 128,
                            temperature: float = 0.8) -> str:
    """
    Generate a single completion for the given HumanEval prompt using HF Transformers.
    Returns the **body only**, correctly indented.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )[0]

    full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return postprocess_completion(prompt, full_text)

if __name__ == "__main__":
    prompt = "def return1():\n"
    print(generate_one_completion(prompt))



problems = read_problems()

num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)