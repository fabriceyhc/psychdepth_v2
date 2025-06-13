import argparse
import subprocess
import os
import sys
import threading
import pandas as pd
from typing import Dict, Any
import json
from datetime import datetime

def summarize_math_evaluation_results(csv_path: str) -> Dict[str, Any]:
    """Reads math evaluation results CSV, calculates accuracy, and returns a summary dict."""
    summary = {
        "raw_data_file": os.path.basename(csv_path) if csv_path else None,
        "accuracy": None, 
        "error": None
    }
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and 'is_correct' in df.columns:
                # Calculate accuracy (ensure is_correct is treated as boolean/int)
                accuracy = df['is_correct'].mean() * 100
                summary["accuracy"] = round(accuracy, 2)
            else:
                summary["error"] = "File is empty or missing 'is_correct' column."
        except Exception as e:
            summary["error"] = f"Error reading file: {e}"
    elif csv_path:
        summary["error"] = "Results file not found."
    else:
        summary["error"] = "Results file path not specified."

    return summary

def summarize_story_results(csv_path: str) -> Dict[str, Any]:
    """Reads Story scores CSV, calculates mean scores, and returns a summary dict."""
    summary = {
        "raw_data_file": os.path.basename(csv_path) if csv_path else None,
        "mean_scores": {}, 
        "error": None
    }
    if csv_path and os.path.exists(csv_path):
        try:
            df_stories = pd.read_csv(csv_path)
            if not df_stories.empty:
                # Identify score columns (ending with '_score')
                score_columns = [col for col in df_stories.columns if col.endswith('_score')]
                if score_columns:
                    # Calculate mean for each score column
                    mean_scores = df_stories[score_columns].mean().to_dict()
                    # Round mean scores for the report
                    summary["mean_scores"] = {k: round(v, 2) for k, v in mean_scores.items()}
                else:
                    summary["error"] = "No score columns found ending with '_score'."
            else:
                summary["error"] = "Results file is empty."
        except Exception as e:
            summary["error"] = f"Error reading file: {e}"
    elif csv_path:
        summary["error"] = "Results file not found."
    else:
        summary["error"] = "Results file path not specified."

    return summary

# Function to read from a pipe and print with a prefix
def _stream_pipe(pipe, prefix=""):
    """Reads lines from a pipe and prints them with a prefix."""

    while True:
        line = pipe.readline()
        if not line:  # Line is empty if pipe is closed
            break
        # Write the line with prefix and flush immediately
        sys.stdout.write(prefix + line)
        sys.stdout.flush()
    pipe.close() 


def run_command(command_args, env=None):
    """
    Runs an external command, streams its output live using threads,
    and checks for errors.
    """
    command_string = ' '.join(command_args)
    print(f"\n>>> Running command: {command_string}")  # Added leading '>>>' and newline
    if env and 'PYTHONPATH' in env:
        print(f"    (with PYTHONPATH={env['PYTHONPATH']})")

    process = None  # Initialize process variable
    stdout_thread = None
    stderr_thread = None

    try:
        process = subprocess.Popen(
            command_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, 
            env=env,  
            bufsize=1 
        )

        # --- Stream Output Live using Threads ---
        # Create separate threads to read stdout and stderr concurrently
        stdout_thread = threading.Thread(target=_stream_pipe, args=(process.stdout, ""))
        stderr_thread = threading.Thread(target=_stream_pipe, args=(process.stderr, "STDERR: "))

        # Start the threads
        stdout_thread.start()
        stderr_thread.start()

        # --- Wait for Process to Finish ---
        return_code = process.wait()

        # --- Ensure all output is read ---
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        if return_code == 0:
            print(f"\n<<< Command finished successfully: {command_string}")
            return True
        else:
            print(f"\n<<< Error: Command exited with non-zero status {return_code}: {command_string}")
            return False

    except FileNotFoundError:
        print(f"\n<<< Error: The executable '{command_args[0]}' was not found.")
        print(f"    Attempted command: {command_string}")
        # Attempt to clean up process and threads if they were started
        if process:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                pass
        if stdout_thread and stdout_thread.is_alive():
            stdout_thread.join(timeout=1)
        if stderr_thread and stderr_thread.is_alive():
            stderr_thread.join(timeout=1)
        return False

    except Exception as e:
        print(f"\n<<< An unexpected error occurred while running {command_string}: {e}")
        # Attempt to clean up process and threads
        if process:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                pass
        if stdout_thread and stdout_thread.is_alive():
            stdout_thread.join(timeout=1)
        if stderr_thread and stderr_thread.is_alive():
            stderr_thread.join(timeout=1)
        return False


def main(args):
    """
    Orchestrates the evaluation and optional training pipeline.
    """
    # --- Path and Variable Setup ---

    base_results_dir = os.path.join(args.save_dir, os.path.basename(args.base_model))
    print(f"Base results directory: {base_results_dir}")
    # Determine the specific directory for evaluation results based on whether training happens
    if args.stage:
        # Training stage specified, evaluation results go into a stage-specific subdir
        eval_save_dir = base_results_dir + "_" + args.stage

        post_train_base_dir = os.path.join("./models", os.path.basename(args.base_model), args.datasets.replace(',', '_'), args.stage)

        output_model_name_part = os.path.basename(args.base_model) + "_" + args.stage
        train_out_dir = os.path.join(post_train_base_dir, "train", output_model_name_part)
        export_out_dir = os.path.join(post_train_base_dir, "export", output_model_name_part)

        trained_model_path = export_out_dir

    else:
        # No training stage specified, evaluating the fine-tuned model
        eval_save_dir = base_results_dir + "_" + os.path.basename(args.fine_tuned_model)
        trained_model_path = args.fine_tuned_model  # Evaluate the provided fine_tuned_model

    # Ensure output directories exist
    os.makedirs(eval_save_dir, exist_ok=True)
    if args.stage:
        # Also create the specific training and export directories
        os.makedirs(train_out_dir, exist_ok=True)
        os.makedirs(export_out_dir, exist_ok=True)

    shots = 0

    # --- Environment Setup for Subprocesses ---
    my_env = os.environ.copy()
    project_root = "."
    current_python_path = my_env.get('PYTHONPATH', '')
    if current_python_path:
        my_env['PYTHONPATH'] = project_root + os.pathsep + current_python_path
    else:
        my_env['PYTHONPATH'] = project_root
    print(f"\nSetting PYTHONPATH for subprocesses: {my_env['PYTHONPATH']}")

    # --- Pipeline Steps ---

    # Evaluate the base model before training
    print(f"\n--- Evaluating Base Model with {args.eval_dataset} ---")
    
    # Select the appropriate math evaluation module based on eval_dataset argument
    math_eval_module = f"evaluate.annotators.{args.eval_dataset}"
        
    math_evaluation_command_base = [
        sys.executable,
        '-m', math_eval_module, 
        "--base_model", args.base_model,
        "--cache_dir", args.cache_dir,
        "--type", args.type,
        "--save_dir", eval_save_dir, 
        "--shots", str(shots) 
    ]
    # if not run_command(math_evaluation_command_base, env=my_env):
    #     print(f"\n{args.eval_dataset} evaluation (base model) failed. Exiting.")
    #     sys.exit(1)
    math_evaluation_script_base_output_csv = os.path.join(eval_save_dir, f"{os.path.basename(args.base_model)}_{shots}shot.csv")

    print("\n--- Generating Stories (Base Model) ---")
    story_gen_module = "dataset.generate_stories"
    story_generation_base_output_file = f"{os.path.basename(args.base_model)}_stories.csv"
    story_generation_base_output_path = os.path.join(eval_save_dir, story_generation_base_output_file)
    story_generation_command_base = [
        sys.executable,
        '-m', story_gen_module,
        "--model_id", args.base_model,
        "--backend_type", args.type,
        "--output_dir", eval_save_dir,
        "--output_csv", story_generation_base_output_file
    ]
    # if not run_command(story_generation_command_base, env=my_env):
    #     print("\nStory generation (base model) failed. Exiting.")
    #     sys.exit(1)

    print("\n--- Evaluating Generated Stories (Base Model) ---")
    story_eval_module = "story_eval.dspy.multiscore.annotate"
    story_evaluation_base_output_file = f"{os.path.basename(args.base_model)}_stories_scored.csv"
    story_evaluation_base_output_path = os.path.join(eval_save_dir, story_evaluation_base_output_file)
    story_evaluation_command_base = [
        sys.executable,
        '-m', story_eval_module, 
        "--dataset", story_generation_base_output_path,
        "--output", story_evaluation_base_output_path, 
    ]
    # if not run_command(story_evaluation_command_base, env=my_env):
    #     print("\nStory evaluation (base model) failed. Exiting.")
    #     sys.exit(1)

    # --- Training Step (Conditional) ---
    if args.stage:
        print(f"\n--- Training Model ({args.stage}) ---")

        # Set up environment for DeepSpeed training
        train_env = my_env.copy()
        train_env['FORCE_TORCHRUN'] = '1'  # Add this environment variable for DeepSpeed

        train_module = "train.writer.run"
        train_command = [
            sys.executable,
            '-m', train_module, 
            "--model_name", args.base_model, 
            "--template", args.template,
            "--datasets", args.datasets, 
            "--stage", args.stage,
            "--train_output_dir", train_out_dir,
            "--export_output_dir", export_out_dir,
            "--num_train_epochs", str(args.num_train_epochs)
        ]

        if not run_command(train_command, env=train_env):
            print("\nTraining failed. Exiting.")
            sys.exit(1)

        print(f"\nTraining finished. Expected exported model path: {trained_model_path}")

    # --- Post-Training Evaluation (or Evaluation of fine_tuned_model) ---
    print(f"\n--- Evaluating Model ({'After Training' if args.stage else 'Fine-tuned'}) with {args.eval_dataset} ---")
    # Evaluate the model after training or use the fine-tuned model specified

    # Use the same math evaluation module that was selected earlier
    math_evaluation_command_post = [
        sys.executable,
        '-m', math_eval_module,
        "--base_model", trained_model_path,
        "--cache_dir", args.cache_dir,
        "--type", args.type,
        "--save_dir", eval_save_dir,
        "--shots", str(shots)
    ]
    if not run_command(math_evaluation_command_post, env=my_env):
        print(f"\n{args.eval_dataset} evaluation (post-training/fine-tuned) failed. Exiting.")
        sys.exit(1)
    math_evaluation_script_post_output_csv = os.path.join(eval_save_dir, f"{os.path.basename(trained_model_path)}_{shots}shot.csv")

    print(f"\n--- Generating Stories ({'After Training' if args.stage else 'Fine-tuned'}) ---")
    story_generation_post_output_file = f"{os.path.basename(trained_model_path)}_stories.csv"
    story_generation_post_output_path = os.path.join(eval_save_dir, story_generation_post_output_file)
    story_generation_command_post = [
        sys.executable,
        '-m', story_gen_module,  
        "--model_id", trained_model_path, 
        "--backend_type", args.type,
        "--output_dir", eval_save_dir, 
        "--output_csv", story_generation_post_output_file 
    ]
    if not run_command(story_generation_command_post, env=my_env):
        print("\nStory generation (post-training/fine-tuned) failed. Exiting.")
        sys.exit(1)

    print(f"\n--- Evaluating Generated Stories ({'After Training' if args.stage else 'Fine-tuned'}) ---")
    story_evaluation_post_output_file = f"{os.path.basename(trained_model_path)}_stories_scored.csv"
    story_evaluation_post_output_path = os.path.join(eval_save_dir, story_evaluation_post_output_file)
    story_evaluation_command_post = [
        sys.executable,
        '-m', story_eval_module,
        "--dataset", story_generation_post_output_path,
        "--output", story_evaluation_post_output_path,
        "--strategy", args.strategy
    ]
    if not run_command(story_evaluation_command_post, env=my_env):
        print("\nStory evaluation (post-training/fine-tuned) failed. Exiting.")
        sys.exit(1)

    # --- Final JSON Report Generation ---
    print("\n--- Generating Final Evaluation Report ---")

    # --- Collect Summaries for Base Model ---
    print("  Summarizing Base Model Results...")
    base_model_name_for_report = os.path.basename(args.base_model)  # Use base name for clarity

    base_evaluation_data = {
        "model_name": base_model_name_for_report,
        "math_evaluation": summarize_math_evaluation_results(math_evaluation_script_base_output_csv),
        "story_evaluation": summarize_story_results(story_evaluation_base_output_path)
    }
    if base_evaluation_data["math_evaluation"].get("error"):
        print(f"    Issue summarizing Base Math: {base_evaluation_data['math_evaluation']['error']}")
    if base_evaluation_data["story_evaluation"].get("error"):
        print(f"    Issue summarizing Base Story: {base_evaluation_data['story_evaluation']['error']}")

    # --- Collect Summaries for Post-Training/Fine-Tuned Model ---
    print("  Summarizing Post-Training/Fine-tuned Model Results...")
    # The model name for the report depends on whether training happened
    post_model_name_for_report = os.path.basename(trained_model_path)

    post_evaluation_data = {
        "model_name": post_model_name_for_report,
        "math_evaluation": summarize_math_evaluation_results(math_evaluation_script_post_output_csv),
        "story_evaluation": summarize_story_results(story_evaluation_post_output_path)
    }
    if post_evaluation_data["math_evaluation"].get("error"):
        print(f"    Issue summarizing Post-eval Math: {post_evaluation_data['math_evaluation']['error']}")
    if post_evaluation_data["story_evaluation"].get("error"):
        print(f"    Issue summarizing Post-eval Story: {post_evaluation_data['story_evaluation']['error']}")

    # --- Structure Final Report Data ---
    final_report_data = {
        "report_metadata": {
            "pipeline_run_model_evaluated": post_model_name_for_report, 
            "base_model_evaluated": base_model_name_for_report,
            "stage_trained": args.stage if args.stage else "none", 
            "fine_tuned_model_used_if_no_training": args.fine_tuned_model if not args.stage else None,
            "evaluation_directory": eval_save_dir, 
            "evaluation_dataset": args.eval_dataset,
            "training_datasets": args.datasets,
            "report_generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "evaluation_results": {
            "base_model_results": base_evaluation_data,
            "post_training_or_fine_tuned_results": post_evaluation_data
        }
    }

    # --- Save Final Report JSON ---
    report_filename = "final_evaluation_report.json"
    report_path = os.path.join(eval_save_dir, report_filename)

    try:
        with open(report_path, 'w') as f:
            json.dump(final_report_data, f, indent=4)  # Use indent for readability
        print(f"\nFinal report saved to {report_path}")
    except Exception as e:
        print(f"Error saving final report to {report_path}: {e}")

    print("--- Final Report Generation Complete ---")

    sys.exit(0) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Evaluation and optional Training pipeline using python -m.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--base_model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Base model identifier or path."
    )
    parser.add_argument(
        "--cache_dir",
        default='/data2/.shared_models',
        help="Directory for storing cached models from Hugging Face."
    )
    parser.add_argument(
        "--eval_dataset",
        required=True,
        choices=["aime", "math500", "gsm8k"],
        help="Dataset identifier for evaluation."
    )
    parser.add_argument(
        "--type",
        default="transformers",
        choices=["transformers", "llama.cpp"],
        help="Model type (transformers or llama.cpp)."
    )
    parser.add_argument(
        "--datasets",
        help="Dataset identifiers for training."
    )
    parser.add_argument(
        "--stage",
        choices=["sft", "dpo", "kto"],
        help="Training stage to perform (sft, dpo, or kto). If not provided, skips training and uses --fine_tuned_model for post-evaluation."
    )
    parser.add_argument(
        "--template",
        help="Prompt template key (e.g., 'llama3', 'chatml') for training. Required if --stage is provided.",
    )
    parser.add_argument(
        "--fine_tuned_model",
        help="Fine-tuned model identifier or path to use for post-evaluation if --stage is NOT provided."
    )
    parser.add_argument(
        "--save_dir",
        default="./pipeline/results",
        help="Base directory for saving evaluation results."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of epochs to train (only used if --stage is provided)."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-70B-Instruct/MIPROv2_Predict-PsychDepthAssessment_handpicked-demos=7_persona.json",
    )

    args = parser.parse_args()

    # Add a check: If stage is provided, template is required
    if args.stage and not args.template:
        parser.error("--template is required when --stage is provided.")

    main(args)