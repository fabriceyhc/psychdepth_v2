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

# Constants
SEED = 0
STORY = "A story to be evaluated for the different components of psychological depth."
PDS_COMPONENT = "The specific component of psychological depth to evaluate in the story."
SCORE = "Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is average and 5 is greatly above average (should be rare to provide this score)."
EXPLANATION = "Optional explanation for the psychological depth score."

class DepthS(dspy.Signature):
    """
    1. Review the given components of psychological depth: authenticity, emotion provoking, empathy, engagement, and narrative complexity. Be sure to understand each concept and the questions that characterize them.
    2. Read a given story, paying special attention to components of psychological depth.
    3. Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is average and 5 is greatly above average (should be rare to provide this score).
    4. Lastly, estimate the likelihood that each story was authored by a human or an LLM. Think about what human or LLM writing characteristics may be. Assign a score from 1 to 5, where 1 means very likely LLM written and 5 means very likely human written. 

    ###Description of Psychological Depth Components:  
    
    We define sychological depth in terms of the following concepts, each illustrated by several questions: 

    - Authenticity 
        - Does the writing feel true to real human experiences? 
        - Does it represent psychological processes in a way that feels authentic and believable? 
    - Emotion Provoking 
        - How well does the writing depict emotional experiences? 
        - Does it explore the nuances of the characters' emotional states, rather than just describing them in simple terms? 
        - Can the writing show rather than tell a wide variety of emotions? 
        - Do the emotions that are shown in the text make sense in the context of the story? 
    - Empathy 
        - Do you feel like you were able to empathize with the characters and situations in the text? 
        - Do you feel that the text led you to introspection, or to new insights about yourself or the world?" 
    - Engagement 
        - Does the text engage you on an emotional and psychological level? 
        - Do you feel the need to keep reading as you read the text? 
    - Narrative Complexity 
        - Do the characters in the story have multifaceted personalities? Are they developed beyond stereotypes or tropes? Do they exhibit internal conflicts? 
        - Does the writing explore the complexities of relationships between characters? 
        - Does it delve into the intricacies of conflicts and their partial or complete resolutions? 
    """
    story: str = dspy.InputField(desc=STORY)
    psychological_depth_component: str = dspy.InputField(desc=PDS_COMPONENT)
    score: float = dspy.OutputField(desc=SCORE)

class DepthSE(dspy.Signature):
    """
    1. Review the given components of psychological depth: authenticity, emotion provoking, empathy, engagement, and narrative complexity. Be sure to understand each concept and the questions that characterize them.
    2. Read a given story, paying special attention to components of psychological depth.
    3. Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is average and 5 is greatly above average (should be rare to provide this score).
    4. Lastly, estimate the likelihood that each story was authored by a human or an LLM. Think about what human or LLM writing characteristics may be. Assign a score from 1 to 5, where 1 means very likely LLM written and 5 means very likely human written. 

    ###Description of Psychological Depth Components:  
    
    We define sychological depth in terms of the following concepts, each illustrated by several questions: 

    - Authenticity 
        - Does the writing feel true to real human experiences? 
        - Does it represent psychological processes in a way that feels authentic and believable? 
    - Emotion Provoking 
        - How well does the writing depict emotional experiences? 
        - Does it explore the nuances of the characters' emotional states, rather than just describing them in simple terms? 
        - Can the writing show rather than tell a wide variety of emotions? 
        - Do the emotions that are shown in the text make sense in the context of the story? 
    - Empathy 
        - Do you feel like you were able to empathize with the characters and situations in the text? 
        - Do you feel that the text led you to introspection, or to new insights about yourself or the world?" 
    - Engagement 
        - Does the text engage you on an emotional and psychological level? 
        - Do you feel the need to keep reading as you read the text? 
    - Narrative Complexity 
        - Do the characters in the story have multifaceted personalities? Are they developed beyond stereotypes or tropes? Do they exhibit internal conflicts? 
        - Does the writing explore the complexities of relationships between characters? 
        - Does it delve into the intricacies of conflicts and their partial or complete resolutions? 
    """
    story: str = dspy.InputField(desc=STORY)
    psychological_depth_component: str = dspy.InputField(desc=PDS_COMPONENT)
    score: float = dspy.OutputField(desc=SCORE)
    explanation: str = dspy.OutputField(desc=EXPLANATION)

class DepthES(dspy.Signature):
    """
    1. Review the given components of psychological depth: authenticity, emotion provoking, empathy, engagement, and narrative complexity. Be sure to understand each concept and the questions that characterize them.
    2. Read a given story, paying special attention to components of psychological depth.
    3. Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is average and 5 is greatly above average (should be rare to provide this score).
    4. Lastly, estimate the likelihood that each story was authored by a human or an LLM. Think about what human or LLM writing characteristics may be. Assign a score from 1 to 5, where 1 means very likely LLM written and 5 means very likely human written. 

    ###Description of Psychological Depth Components:  
    
    We define sychological depth in terms of the following concepts, each illustrated by several questions: 

    - Authenticity 
        - Does the writing feel true to real human experiences? 
        - Does it represent psychological processes in a way that feels authentic and believable? 
    - Emotion Provoking 
        - How well does the writing depict emotional experiences? 
        - Does it explore the nuances of the characters' emotional states, rather than just describing them in simple terms? 
        - Can the writing show rather than tell a wide variety of emotions? 
        - Do the emotions that are shown in the text make sense in the context of the story? 
    - Empathy 
        - Do you feel like you were able to empathize with the characters and situations in the text? 
        - Do you feel that the text led you to introspection, or to new insights about yourself or the world?" 
    - Engagement 
        - Does the text engage you on an emotional and psychological level? 
        - Do you feel the need to keep reading as you read the text? 
    - Narrative Complexity 
        - Do the characters in the story have multifaceted personalities? Are they developed beyond stereotypes or tropes? Do they exhibit internal conflicts? 
        - Does the writing explore the complexities of relationships between characters? 
        - Does it delve into the intricacies of conflicts and their partial or complete resolutions? 
    """
    story: str = dspy.InputField(desc=STORY)
    psychological_depth_component: str = dspy.InputField(desc=PDS_COMPONENT)
    explanation: str = dspy.OutputField(desc=EXPLANATION)
    score: float = dspy.OutputField(desc=SCORE)


def score_pds(example, prediction, trace=None):
    error = abs(prediction.score - example.score)
    # Normalize the MAE so that 0 error maps to 1 accuracy and maximum error maps to 0.
    max_error = 4  # Since the scores range from 1 to 5
    accuracy = 1 - (error / max_error)
    return accuracy


# Dataset Preparation
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

# Evaluation Helper
def evaluate_model(evaluator, model, signature_name, module_name, num_demos, trainset, model_id):
    pre_score = evaluator(model, metric=score_pds)
    
    optimizer = dspy.MIPROv2(
        metric=score_pds,
        num_threads=24,
        max_labeled_demos=num_demos,
        max_bootstrapped_demos=num_demos
    )
    
    optimized_model = optimizer.compile(model, trainset=trainset, requires_permission_to_run=False)
    post_score = evaluator(optimized_model, metric=score_pds)

    save_path = f'./story_eval/dspy/singlescore/optimized_prompts/{model_id}/MIPROv2_{module_name}-{signature_name}_demos={num_demos}.json'
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

# Main Execution
def main(model_id):
    # Server setup
    server_process, port = launch_server_cmd(
        f"python -m sglang.launch_server --model-path {model_id} --download-dir /data2/.shared_models"
    )
    wait_for_server(f"http://localhost:{port}")
    print(f"SGLang server started on http://localhost:{port}")

    try:
        # Data loading
        trainset = prepare_dataset("./data/stories_w_human_annotations_singlescore_train.csv")
        testset  = prepare_dataset("./data/stories_w_human_annotations_singlescore_test.csv", max_rows=200)

        print(testset)

        # Model setup
        lm = dspy.LM(
            f"openai/{model_id}",
            api_base=f"http://localhost:{port}/v1",
            api_key="local",
            model_type='chat'
        )
        dspy.configure(lm=lm)

        MODULES = [dspy.Predict, dspy.ChainOfThought]
        SIGNATURES = [DepthS, DepthSE, DepthES]
        NUM_DEMOS_OPTIONS = [0, 3, 5, 10]

        # Experiment loop
        results = []
        evaluator = Evaluate(devset=testset, num_threads=1, display_progress=True)
        
        for signature in SIGNATURES:
            signature_name = signature.__name__
            
            for module in MODULES:
                model = module(signature)
                module_name = module.__name__
                
                for num_demos in NUM_DEMOS_OPTIONS:
                    result = evaluate_model(evaluator, model, signature_name, module_name, num_demos, trainset, model_id)
                    results.append(result)

        # Ensure summary save directory exists
        summary_dir = f"./story_eval/dspy/singlescore/optimized_prompts/{model_id}"
        os.makedirs(summary_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(f"{summary_dir}/summary.csv")
        terminate_process(server_process)

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        terminate_process(server_process)

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0 python -m story_eval.dspy.singlescore.annotate --model_id meta-llama/Llama-3.1-8B-Instruct

    parser = argparse.ArgumentParser(
        description="Evaluate a model using SGLang for singlescore tasks with a specified Hugging Face model_id."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model id to use (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    args = parser.parse_args()
    main(args.model_id)