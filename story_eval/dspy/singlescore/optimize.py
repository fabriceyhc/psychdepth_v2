import random
import os
import dspy
import pandas as pd
import textwrap
import traceback
import argparse
from dspy.datasets import DataLoader
from dspy.evaluate import Evaluate
from sglang.utils import launch_server_cmd, wait_for_server, print_highlight, terminate_process

from story_eval.dspy.singlescore.signatures import PDSSinglescoreS, PDSSinglescoreSE, PDSSinglescoreES

SEED = 0

DEFAULT_PERSONAS = [
    "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you specialize in evaluating the genuineness and believability of characters, dialogue, and scenarios in stories.",
    "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you focus on identifying and assessing moments in the narrative that effectively evoke empathetic connections with the characters.",
    "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you evaluate how well a story captures and maintains the reader's interest through pacing, suspense, and narrative flow.",
    "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you examine the text for its ability to provoke a wide range of intense emotional responses in the reader.",
    "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you analyze the structural and thematic intricacy of the plot, character development, and the use of literary devices.",
]

class MultiPersonaModule(dspy.Module):
    def __init__(self, base_model, personas):
        super().__init__()
        self.base_model = base_model
        self.personas = personas

    def forward(self, story, psychological_depth_component):
        scores = []
        for persona in self.personas:
            modified_story = f"System Prompt: {persona}\n\nStory: {story}"
            prediction = self.base_model(story=modified_story, psychological_depth_component=psychological_depth_component)
            scores.append(prediction.score)
        avg_score = sum(scores) / len(scores)
        return dspy.Prediction(score=avg_score)

def score_pds(example, prediction, trace=None):
    error = abs(prediction.score - example.score)
    max_error = 4
    accuracy = 1 - (error / max_error)
    return accuracy

def prepare_dataset(file_path, max_rows=None):
    df = pd.read_csv(file_path)
    dataset = [
        dspy.Example(
            story=row["text"],
            psychological_depth_component=row["pds_component"],
            score=row["score"]
        ).with_inputs("story", "psychological_depth_component")
        for _, row in df.iterrows()
    ]
    random.Random(SEED).shuffle(dataset)
    return dataset[:max_rows] if max_rows else dataset

def evaluate_model(evaluator, model, signature_name, module_name, num_demos, trainset, model_id, personas):
    # Pre-evaluation with personas
    if personas:
        pre_model = MultiPersonaModule(model, personas)
    else:
        pre_model = model
    pre_score = evaluator(pre_model, metric=score_pds)
    
    # Optimize base model without personas
    optimizer = dspy.MIPROv2(
        metric=score_pds,
        num_threads=24,
        max_labeled_demos=num_demos,
        max_bootstrapped_demos=num_demos
    )
    optimized_model = optimizer.compile(model, trainset=trainset, requires_permission_to_run=False)
    
    # Post-evaluation with personas
    if personas:
        post_model = MultiPersonaModule(optimized_model, personas)
    else:
        post_model = optimized_model
    post_score = evaluator(post_model, metric=score_pds)

    save_path = f'./story_eval/dspy/singlescore/optimized_prompts/{model_id}/MIPROv2_{module_name}-{signature_name}_demos={num_demos}{"_persona" if personas else ""}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    optimized_model.save(save_path, save_program=False)
    
    return {
        "module": module_name,
        "signature": signature_name,
        "num_demos": num_demos,
        "pre_score": pre_score,
        "post_score": post_score,
        "save_path": save_path,
    }

def main(model_id, use_personas):
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    num_gpus = 1
    if cuda_visible_devices:
        gpu_list = [gpu.strip() for gpu in cuda_visible_devices.split(",") if gpu.strip()]
        num_gpus = len(gpu_list)

    server_cmd = f"python -m sglang.launch_server --model-path {model_id} --download-dir /data2/.shared_models/hf --tp {num_gpus}"
    server_process, port = launch_server_cmd(server_cmd)
    wait_for_server(f"http://localhost:{port}")

    try:
        trainset = prepare_dataset("./data/stories_w_human_annotations_singlescore_train.csv")
        testset = prepare_dataset("./data/stories_w_human_annotations_singlescore_test.csv", max_rows=200)

        lm = dspy.LM(
            f"openai/{model_id}",
            api_base=f"http://localhost:{port}/v1",
            api_key="local",
            model_type='chat'
        )
        dspy.configure(lm=lm)

        MODULES = [dspy.Predict, dspy.ChainOfThought]
        SIGNATURES = [PDSSinglescoreS, PDSSinglescoreSE, PDSSinglescoreES]
        NUM_DEMOS_OPTIONS = [0, 3, 5, 10]
        personas = DEFAULT_PERSONAS if use_personas else []

        results = []
        evaluator = Evaluate(devset=testset, num_threads=1, display_progress=True)
        
        for signature in SIGNATURES:
            signature_name = signature.__name__
            for module in MODULES:
                model = module(signature)
                module_name = module.__name__
                for num_demos in NUM_DEMOS_OPTIONS:
                    result = evaluate_model(evaluator, model, signature_name, module_name, num_demos, trainset, model_id, personas)
                    results.append(result)

        summary_dir = f"./story_eval/dspy/singlescore/optimized_prompts/{model_id}"
        os.makedirs(summary_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(f"{summary_dir}/summary_persona.csv" if use_personas else f"{summary_dir}/summary.csv")
        terminate_process(server_process)

    except Exception as e:
        traceback.print_exc()
        terminate_process(server_process)

if __name__ == "__main__":
    
    # CUDA_VISIBLE_DEVICES=0 python -m story_eval.dspy.singlescore.optimize --model_id meta-llama/Llama-3.1-8B-Instruct --use_personas

    parser = argparse.ArgumentParser(description="Evaluate a model with personas using SGLang.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--use_personas", action="store_true", help="Enable persona-based evaluation")
    args = parser.parse_args()
    main(args.model_id, args.use_personas)