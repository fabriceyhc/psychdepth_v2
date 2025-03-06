import random
import os
import dspy
import pandas as pd
import textwrap
import traceback
import argparse
from sglang.utils import launch_server_cmd, wait_for_server, print_highlight, terminate_process

# Constants
SEED = 0
STORY = "A story to be evaluated for the different components of psychological depth."
PDS_COMPONENT = "The specific component of psychological depth to evaluate in the story."
SCORE = "Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is average and 5 is greatly above average (should be rare to provide this score)."
EXPLANATION = "Optional explanation for the psychological depth score."

class PsychDepthAssessment(dspy.Signature):
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

    story: str = dspy.InputField(desc="A story to be evaluated for psychological depth.")
    authenticity_score: float = dspy.OutputField(desc="1=Unrealistic, 5=Profoundly authentic")
    emotion_provoking_score: float = dspy.OutputField(desc="1=Flat, 5=Deeply moving")
    empathy_score: float = dspy.OutputField(desc="1=Detached, 5=Transformative empathy")
    engagement_score: float = dspy.OutputField(desc="1=Boring, 5=Irresistibly compelling")
    narrative_complexity_score: float = dspy.OutputField(desc="1=Shallow, 5=Masterfully complex")
    human_likeness_score: float = dspy.OutputField(desc="1=Clearly AI, 5=Undeniably human")

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

def load_and_recreate_strategy(strategy_file):
    """
    Given a saved dspy program JSON file (the optimized prompt), recreate the
    dspy program instance based on the file name and load the JSON.
    
    Expected file name format:
    "MIPROv2_<ModuleType>-<Signature>_demos=<num>.json"
    For example: "MIPROv2_Predict-PsychDepthAssessment_demos=10.json"
    """
    SIGNATURES = {"PsychDepthAssessment": PsychDepthAssessment, "DepthS": DepthS, "DepthSE": DepthSE, "DepthES": DepthES}
    filename = os.path.basename(strategy_file)
    parts = filename.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected file name format: {filename}")
    # The second part should be in the form "Predict-PsychDepthAssessment" (or ChainOfThought)
    module_and_sig = parts[1].split("-")
    if len(module_and_sig) < 2:
        raise ValueError(f"Unexpected module/signature format in: {filename}")
    module_type = module_and_sig[0]  # e.g. "Predict" or "ChainOfThought"
    signature = module_and_sig[1] # e.g. "PsychDepthAssessment" or "DepthS"

    # Recreate the dspy program accordingly.
    if module_type == "Predict":
        program_instance = dspy.Predict(SIGNATURES[signature])
    elif module_type == "ChainOfThought":
        program_instance = dspy.ChainOfThought(SIGNATURES[signature])
    else:
        raise ValueError(f"Unknown module type: {module_type} in file {filename}")
    
    # Load the saved JSON into the newly created instance.
    program_instance.load(strategy_file)
    return program_instance


def prepare_dataset(evaluation_type):
    """
    Loads the test CSV dataset and creates a list of dspy.Example objects.
    
    We use the "text" column for evaluation and preserve the "story_id" for merging later.
    """
    dataset = []
    if (evaluation_type == "multiscore"):
        df = pd.read_csv("./data/stories_w_human_annotations_multiscore.csv")
        for _, row in df.iterrows():
            example = dspy.Example(story=row["text"]).with_inputs("story")
            # Save the story_id as an attribute for later merging.
            example.story_id = row["story_id"]
            dataset.append(example)

    else:
        df = pd.read_csv("./data/stories_w_human_annotations_singlescore.csv")
        df_unique = df.drop_duplicates(subset=["story_id", "pds_component"])
        print(f"Number of unique stories: {df_unique.shape[0]}")
        dataset = [
            dspy.Example(
                story=row["text"],
                psychological_depth_component=row["pds_component"],
                score=row["score"],
                story_id=row["story_id"]
            ).with_inputs("story", "psychological_depth_component")
            for _, row in df_unique.iterrows()
        ]
    return dataset

def get_top_N_strategies(evaluation_type, N):
    """
    Scans the directory ./story_eval/dspy/<evaluation_type>/optimized_prompts
    for summary.csv files, loads them, and returns the top N rows (sorted by 'post_score')
    across all found summaries.

    Directory structure example:
        ./story_eval/dspy/<evaluation_type>/optimized_prompts/<some_model_id>/summary.csv

    :param evaluation_type: "multiscore" or "singlescore", etc.
    :param N: number of top strategies to pick overall
    :return: a DataFrame containing the top N strategies (by 'post_score') 
             from all summary.csv files under the given evaluation_type directory
    """
    # summary_dir = f"./story_eval/dspy/{evaluation_type}"
    summary_dir = f"./psychdepth_v2/story_eval/dspy/{evaluation_type}/optimized_prompts/meta-llama"
    summary_paths = {}
    # Recursively walk the summary_dir to find any summary.csv files
    for root, dirs, files in os.walk(summary_dir):
        for file in files:
            if file == "summary.csv":
                full_path = os.path.join(root, file)
                parts = os.path.normpath(root).split(os.sep)
                # Assume model id is contained in the last 2 parts of the path
                model_id = "/".join(parts[-2:])
                summary_paths[model_id] = full_path

    all_summaries = []
    for model in summary_paths:
        summary = pd.read_csv(summary_paths[model])
        summary['model_id'] = model  # add model_id info to each row
        all_summaries.append(summary)
    all_summaries = pd.concat(all_summaries, ignore_index=True)
    return all_summaries.nlargest(N, 'post_score')

def evaluate_strategy_on_dataset(strategy, dataset, evaluation_type, signature):
    """
    Iterates through the dataset, applying the given strategy to each example.
    Returns a list of prediction dictionaries.
    """
    predictions = []
    if evaluation_type == "multiscore":
        for example in dataset:
            try:
                prediction = strategy(**example.inputs())
                result = {
                    "story_id": example.story_id,
                    "authenticity_score": prediction.authenticity_score,
                    "emotion_provoking_score": prediction.emotion_provoking_score,
                    "empathy_score": prediction.empathy_score,
                    "engagement_score": prediction.engagement_score,
                    "narrative_complexity_score": prediction.narrative_complexity_score,
                    "human_likeness_score": prediction.human_likeness_score
                }
            except Exception as e:
                print("Error processing example:", e)
                traceback.print_exc()
                result = {"story": example.story, "error": str(e)}
            predictions.append(result)
    else:
        for example in dataset:
            try:
                prediction = strategy(**example.inputs())
                result = {
                    "story_id": example.story_id,
                    "psychological_depth_component": example.psychological_depth_component,
                    "score": prediction.score,
                    }
            except Exception as e:
                print("Error processing example:", e)
                traceback.print_exc()
                result = {"story": example.story, "error": str(e)}
            predictions.append(result)
    return predictions

def main(top_N, evaluation_type, strategy):
    strategies_by_model = {}
    if strategy != None:
        parts = os.path.normpath(strategy).split(os.sep)
        # Assume model id is contained in the last 2 parts of the path
        # parse the model_id and evaluation_type from the given strategy path
        model_id = "/".join(parts[-2:])
        evaluation_type = parts[2]
        strategies_by_model[model_id] = strategy
    else: 
        # Get top-N strategies across all models
        top_strategies_df = get_top_N_strategies(evaluation_type, top_N)
        # Group save paths by model_id
        strategies_by_model = {}
        for _, row in top_strategies_df.iterrows():
            save_path = row["save_path"]
            model_id = row["model_id"]
            signature = row["signature"]
            strategies_by_model.setdefault(model_id, []).append(save_path)
    
    print(strategies_by_model)
    # Load test dataset
    testset = prepare_dataset(evaluation_type)
    
    # For each model id from the best strategies, launch one SGLang server and evaluate strategies.
    for model_id, strategy_files in strategies_by_model.items():
        try: 
            # Determine GPU count
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            num_gpus = 1
            if cuda_visible_devices:
                # Count the non-empty entries (in case of stray commas)
                gpu_list = [gpu.strip() for gpu in cuda_visible_devices.split(",") if gpu.strip()]
                num_gpus = len(gpu_list)
            
            # Server setup with optional --tp argument
            server_cmd = f"python -m sglang.launch_server --model-path {model_id} --download-dir /data2/.shared_models/hf --tp {num_gpus}"
            server_process, port = launch_server_cmd(server_cmd)
            wait_for_server(f"http://localhost:{port}")
            print(f"SGLang server started on http://localhost:{port}")
            # Setup the language model using the extracted model_id

            lm = dspy.LM(
                f"openai/{model_id}",
                api_base=f"http://localhost:{port}/v1",
                api_key="local",
                model_type='chat'
            )
            dspy.configure(lm=lm)
            
            for strategy_file in strategy_files:
                try:
                    print(f"Evaluating strategy from file: {strategy_file}")
                    loaded_program = load_and_recreate_strategy(strategy_file)
                    predictions = evaluate_strategy_on_dataset(loaded_program, testset, evaluation_type, signature)
                    df_predictions = pd.DataFrame(predictions)
                    # Save predictions using a filename derived from the strategy file.
                    out_filename = f"{model_id.replace('/', '_')}_predictions_" + os.path.basename(strategy_file).split('.')[0] + ".csv"
                    out_path = os.path.join("./story_eval/dspy/dspy_annotations/", out_filename)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    df_predictions.to_csv(out_path, index=False)
                    print(f"Predictions saved to {out_path}")
                except Exception as e:
                    print(f"Error evaluating strategy {strategy_file}: {e}")
                    traceback.print_exc()
            
            terminate_process(server_process)
        except Exception as e:
            # Use traceback to print the full error details
            print("An error occurred:")
            traceback.print_exc()  # This prints the full stack trace
            terminate_process(server_process)

if __name__ == "__main__":
    # Example usage:
    # Use this command for getting the top N strategies in a given evaluation type
    # CUDA_VISIBLE_DEVICES=2,3 python -m story_eval.dspy.annotate_dspy --top_N 3 --evaluation_type multiscore
    # OR 
    # Use this command for directly loading a desired strategy
    # CUDA_VISIBLE_DEVICES=2,3 python -m story_eval.dspy.annotate_dspy --strategy ./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-70B-Instruct/MIPROv2_Predict-PsychDepthAssessment_demos=10.json
    parser = argparse.ArgumentParser(
        description="Load optimized prompts into a DSPy model and evaluate a dataset using the top strategies."
    )
    parser.add_argument("--top_N", type=int, default=1,
                        help="Number of top strategies to evaluate (default: 1)")
    parser.add_argument("--evaluation_type", type=str, choices=["multiscore", "singlescore"],
                        default="multiscore", help="Evaluation type to run (default: singlescore)")
    parser.add_argument("--strategy", type=str,
                        default=None, help="Annotate the testset using a given strategy")
    args = parser.parse_args()
    main(args.top_N, args.evaluation_type, args.strategy)
