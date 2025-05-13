import time
import re
import pandas as pd
from typing import Dict, List
import guidance
from guidance import gen, user, assistant
from datasets import load_from_disk
from .utils.math_util import compute_score
import argparse
from pathlib import Path
from evaluate.annotators._base import BaseDatasetProcessor


class CodeForcesProcessor(BaseDatasetProcessor):
    """Processor for AIME dataset"""
    
    def load_dataset(self) -> pd.DataFrame:
        # dataset = load_from_disk("./data/open-r1/codeforces")["test"].to_pandas()
        root = Path("data/open-r1/codeforces")
        dataset = load_from_disk(root / "test").to_pandas()
        print(dataset.iloc[0])
        return dataset

    def _is_processed(self, row: Dict, existing: pd.DataFrame) -> bool:
        """Check if row already exists in saved results"""
        if 'id' in row and 'id' in existing.columns:
            return (existing["id"] == row["id"]).any()
        return False


    def grade_answer(self, predicted_answer, ground_truth) -> bool:
        """Grade the predicted answer"""
        return compute_score(predicted_answer, ground_truth)

    def process_entry(self, row: Dict) -> Dict:
        # -------- build a complete problem statement -------------------- #
        parts: List[str] = [f"# {row['title']}\n", row["description"]]
        for tag in ("input_format", "output_format", "examples", "note"):
            if isinstance(row.get(tag), str) and row[tag].strip():
                header = tag.replace("_", " ").title()
                parts.append(f"\n## {header}\n{row[tag]}")
        problem_text = "\n".join(parts).strip()

        # -------- query the model --------------------------------------- #
        start_time = time.time()
        output = self.model + self.annotation_prompt(problem=problem_text)
        time_taken = time.time() - start_time

        # -------- extract the model’s declared final answer ------------- #
        raw_answer = output["answer"]
        answer = raw_answer.partition("Final Answer:")[-1]
        answer = answer.split("\n")[0].strip()
        answer = re.sub(r"[`\*_\u200b]", "", answer)  # remove MD & zero-width chars
        output.set("answer", answer)

        # -------- grade versus reference generation --------------------- #
        is_correct = self.grade_answer(answer, row["generation"])

        # -------- return record ----------------------------------------- #
        return {
            "id": row["id"],
            "problem": problem_text,
            "solution": output["solution"],        # model chain-of-thought
            "predicted_answer": answer,
            "reference_generation": row["generation"],
            "is_correct": is_correct,
            "time_taken": time_taken,
        }
    
    @guidance(dedent=True)
    def annotation_prompt(self, lm, problem: str):
        with user():
            lm += (
                "You are solving a competitive-programming problem. "
                "Reason step by step and put your final answer within \\boxed{}.\n\n"
                f"{problem}"
            )
        with assistant():
            lm += "Step-by-step Solution:\n" + gen(
                name="solution", stop=self.STOP_STRINGS, max_tokens=1200
            )
            lm += "\nFinal Answer:\n" + gen(
                name="answer", stop=self.STOP_STRINGS, max_tokens=200
            )
        return lm
        

def main(args):
    # Initialize the CodeforcesProcessor with the provided configuration
    config = {
        "model": {
            "type": args.type,
            "path": args.base_model,
            "cache_dir": args.cache_dir
        },
        "save_dir": args.save_dir,
        "shots": args.shots
    }
    
    math_processor = CodeForcesProcessor(config)
    math_results = math_processor.run()

    # Compute accuracy
    accuracy = math_results['is_correct'].mean() * 100
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m evaluate.annotators.codeforces

    parser = argparse.ArgumentParser(description="Run codeforces evaluation")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B",
                        help="Base model identifier or path")
    parser.add_argument("--cache_dir", default='data2/.shared_models',
                        help="Directory for storing base models")
    parser.add_argument("--type", default="transformers",
                        choices=["transformers", "llama.cpp"],
                        help="Model type (transformers or llama.cpp)")
    parser.add_argument("--save_dir", default="./evaluate/results/codeforces",
                        help="Directory for saving results")
    parser.add_argument("--shots", type=int, default=0,
                        help="Number of shots for zero-shot evaluation")

    args = parser.parse_args()
    main(args)
