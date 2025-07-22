import argparse, json, re, os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_problems(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def postprocess(prompt: str, full: str) -> str:
    """Strip echoed prompt & keep only the body (indented)."""
    # 1) Remove prompt if it's at the start
    if full.startswith(prompt):
        body = full[len(prompt):]
    else:
        # 2) Heuristic: keep everything after first newline
        body = full.split("\n", 1)[-1]
    # 3) Stop at first fenced code block or two blank lines
    body = re.split(r"\n\s*\n|```", body)[0]
    # Ensure trailing newline
    return body if body.endswith("\n") else body + "\n"

def generate(model, tokenizer, prompt, max_new=192, temperature=0.8, top_p=0.9):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **input_ids,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)



def main(args):
    problems = load_problems(args.problem_file)

    # Load model + tokenizer once
    print("Loading model…")
    tokenizer = AutoTokenizer.from_pretrained(
        "jyc0325/Qwen2.5-1.5B-Instruct-SFT-code",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "jyc0325/Qwen2.5-1.5B-Instruct-SFT-code",
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        device_map="auto" if args.device.startswith("cuda") else None,
        trust_remote_code=True
    )

    os.makedirs(os.path.dirname(args.sample_file), exist_ok=True)
    with open(args.sample_file, "w") as fout:
        for task in tqdm(problems, desc="Generating"):
            prompt = task["prompt"]
            for _ in range(args.num_samples):
                full = generate(model, tokenizer, prompt,
                                max_new=args.max_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p)
                body = postprocess(prompt, full)
                fout.write(json.dumps({
                    "task_id": task["task_id"],
                    "completion": body
                }) + "\n")

    print(f"Wrote samples to {args.sample_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--problem_file", default="data/example_problem.jsonl")
    p.add_argument("--sample_file",  default="data/example_samples.jsonl")
    p.add_argument("--num_samples",  type=int, default=1,
                   help="N completions per task (pass@k).")
    p.add_argument("--device",       default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max_tokens",   type=int, default=192)
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--top_p",        type=float, default=0.9)
    main(p.parse_args())
