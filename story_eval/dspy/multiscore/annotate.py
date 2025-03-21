import random
import os
import dspy
import pandas as pd
import textwrap
import traceback
import argparse
from sglang.utils import launch_server_cmd, wait_for_server, print_highlight, terminate_process
from story_eval.dspy.multiscore.singatures import PDSMultiScore
from story_eval.dspy.multiscore.optimize import MultiPersonaModule, DEFAULT_PERSONAS

def load_and_recreate_strategy(strategy_file):
    filename = os.path.basename(strategy_file)
    parts = filename.split("_")

    module_and_sig = parts[1].split("-")
    module_type = module_and_sig[0]
    persona = parts[-1]

    if persona == "persona.json":
        print("yess")
        if module_type == "Predict":
            program_instance = MultiPersonaModule(dspy.Predict(PDSMultiScore), DEFAULT_PERSONAS)
        elif module_type == "ChainOfThought":
            program_instance = MultiPersonaModule(dspy.ChainOfThought(PDSMultiScore), DEFAULT_PERSONAS)
        else:
            raise ValueError(f"Unknown module type: {module_type} in file {filename}")
        program_instance.base_model.load(strategy_file)
    else:
        if module_type == "Predict":
            program_instance = dspy.Predict(PDSMultiScore)
        elif module_type == "ChainOfThought":
            program_instance = dspy.ChainOfThought(PDSMultiScore)
        else:
            raise ValueError(f"Unknown module type: {module_type} in file {filename}")
        program_instance.load(strategy_file)
    return program_instance

def prepare_dataset(dataset):
    df = pd.read_csv(dataset)
    df = df.drop_duplicates(subset=["story_id"])
    dataset = []
    for _, row in df.iterrows():
        ex = dspy.Example(story=row["text"]).with_inputs("story")
        ex.story_id = row["story_id"]
        dataset.append(ex)
    return dataset

def evaluate_strategy_on_dataset(strategy, dataset):
    predictions = []
    for example in dataset:
        try:
            print(f"Evaluating story: {example.story_id}")
            prediction = strategy(**example.inputs())
            predictions.append({
                "story_id": example.story_id,
                "authenticity_score": prediction.authenticity_score,
                "emotion_provoking_score": prediction.emotion_provoking_score,
                "empathy_score": prediction.empathy_score,
                "engagement_score": prediction.engagement_score,
                "narrative_complexity_score": prediction.narrative_complexity_score,
                "human_likeness_score": prediction.human_likeness_score
            })
        except Exception as e:
            print("Error processing example:", e)
            traceback.print_exc()
            predictions.append({"story_id": example.story_id, "error": str(e)})
    return predictions

def main(dataset, strategy_file):
    testset = prepare_dataset(dataset)
    parts = os.path.normpath(strategy_file).split(os.sep)
    model_id = "/".join(parts[-3:-1])
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
        
        try:
            print(f"Evaluating strategy from file: {strategy_file}")
            loaded_program = load_and_recreate_strategy(strategy_file)
            predictions = evaluate_strategy_on_dataset(loaded_program, testset)
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
    # Use this command for directly loading a desired strategy
    # CUDA_VISIBLE_DEVICES=0,2,3,5 python -m story_eval.dspy.multiscore.annotate --strategy ./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-70B-Instruct/MIPROv2_Predict-PsychDepthAssessment_handpicked-demos=7_persona.json
    parser = argparse.ArgumentParser(
        description="Load optimized prompts into a DSPy model and evaluate a dataset using the top strategies."
    )
    parser.add_argument("--dataset", type=str,
                        default= "./data/stories_w_human_annotations_multiscore.csv", help="Test dataset used for annotation.")
    parser.add_argument("--strategy", type=str,
                        default="./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-70B-Instruct/MIPROv2_Predict-PsychDepthAssessment_handpicked-demos=7_persona.json", help="Annotate the testset using a given strategy")
    args = parser.parse_args()
    main(args.dataset, args.strategy)