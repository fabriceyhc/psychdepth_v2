import sys
import subprocess
import threading
import os
import pandas as pd
import textwrap
import argparse
from pathlib import Path



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

# … everything above is unchanged …

def main(args):
    strategies = [
        "./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PsychDepthAssessment_kmean-demos=10_participant=2_persona.json",
        "./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PsychDepthAssessment_kmean-demos=10_participant=3_persona.json",
        "./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PsychDepthAssessment_kmean-demos=10_participant=4_persona.json",
        "./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PsychDepthAssessment_kmean-demos=10_participant=6_persona.json",
        "./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PsychDepthAssessment_kmean-demos=10_participant=7_persona.json"
    ]

    dataset = args.dataset
    # If user didn’t pass --output, put CSVs next to the dataset file
    base_out_path = args.output or os.path.splitext(dataset)[0] + "_ann"

    generated_csvs = []                     # <‑‑ collect results here
    

    for strategy in strategies:
        print(f"Evaluating using strategy: {strategy}")

        # one CSV per participant strategy
        out_csv = f"{base_out_path}_{os.path.basename(strategy).split('.')[0]}.csv"
        print(out_csv)
        generated_csvs.append(out_csv)

        # ---------- FIX: actually pass --output to the subprocess ----------
        run_command([
            "python", "-m", "story_eval.dspy.multiscore.annotate",
            "--dataset", dataset,
            "--strategy", strategy,
            "--output", out_csv
        ])

    # --------------------------------------------------------------------
    #            Aggregate any CSV whose filename contains “participant”
    # --------------------------------------------------------------------
    participant_csvs = [p for p in generated_csvs
                        if "participant" in os.path.basename(p).lower()
                        and os.path.exists(p)]

    if not participant_csvs:
        print("\n<<< No participant CSVs were produced — nothing to aggregate.")
        return

    print("\n>>> Aggregating the following CSVs:")
    for p in participant_csvs:
        print("    •", p)

    dfs = [pd.read_csv(p) for p in participant_csvs]
    aggregated = (
        pd.concat(dfs, ignore_index=True)
          .groupby("story_id", as_index=False)
          .mean(numeric_only=True)
          .round(3)
    )

    agg_csv = f"{base_out_path}_aggregated_participant.csv"
    aggregated.to_csv(agg_csv, index=False)
    print(f"\n<<< Wrote aggregated results to {agg_csv}\n")


if __name__ == "__main__":
    # Example usage:
    parser = argparse.ArgumentParser(
        description="Load optimized prompts into a DSPy model and evaluate a dataset using the top strategies."
    )
    parser.add_argument("--dataset", type=str,
                        default= "./data/multiscore/stories_w_human_annotations_multiscore.csv", help="Test dataset used for annotation")
    parser.add_argument("--strategy", type=str,
                        default="./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PsychDepthAssessment_kmean-demos=5__participant=3.json", help="Annotate the testset using a given strategy")
    parser.add_argument("--output", type=str,
                        default=None, help="Path of annotation output")
    args = parser.parse_args()
    main(args)