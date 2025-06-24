import time
import pandas as pd
from typing import List, Dict
import guidance
from guidance import models, gen, select, user, system, assistant
from datasets import load_dataset
from .utils.math_util import compute_score
import argparse
import re

from ._base import BaseDatasetProcessor

class MATH500Processor(BaseDatasetProcessor):
    """Processor for MATH-500 dataset"""
    
    def load_dataset(self) -> pd.DataFrame:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test").to_pandas()[:100]
        return dataset

    def _is_processed(self, row: Dict, existing: pd.DataFrame) -> bool:
        """Check if row already exists in saved results"""
        if 'unique_id' in row and 'unique_id' in existing.columns:
            return (existing["unique_id"] == row["unique_id"]).any()
        return False

    def grade_answer(self, predicted_answer, ground_truth) -> bool:
        """Grade the predicted answer"""
        return compute_score(predicted_answer, ground_truth)

    def create_solution_prompt(self, problem: str) -> str:
        """Create a prompt for generating the step-by-step solution"""
        # Base prompt with no examples
        if not self.examples or self.shots == 0:

            prompt = f"""Below is a math question. I want you to reason through the steps and then give a final answer. Your final answer should be in \boxed{{}}.

            Question: {problem}

            Step-by-step Solution:"""
            return prompt
        
        # Add few-shot examples if available
        prompt = "Please solve the following math problems step by step.\n\n"
        
        # Add examples based on the number of shots
        for i in range(min(self.shots, len(self.examples))):
            example = self.examples.iloc[i]
            prompt += f"Problem: {example['problem']}\n\n"
            prompt += f"Step-by-step Solution:\n{example['solution']}\n\n"
        
        # Add the current problem
        prompt += f"Problem: {problem}\n\nStep-by-step Solution:"
        return prompt

    def create_answer_prompt(self, problem: str, solution: str) -> str:
        """Create a prompt for generating the final answer"""
        prompt = f"""Problem: {problem}

        Step-by-step Solution:
        {solution}

        Put your final answer within \\boxed{{}}. Final Answer:"""
        return prompt

    def process_entry(self, row: Dict) -> Dict:
        # 1. First turn: Generate step-by-step solution
        solution_prompt = self.create_solution_prompt(row['problem'])
        
        # Create stop strings that include "Final Answer:" to stop before it
        STOP_SOLUTION = self.STOP_STRINGS + ["Final Answer:", "The final answer is"]
    
        start_time = time.time()
        solution = self.generate_text(
            solution_prompt, 
            max_new_tokens=1024, 
            stop_strings=STOP_SOLUTION
        )
        
        # 2. Second turn: Generate final answer
        answer_prompt = self.create_answer_prompt(row['problem'], solution)

        STOP_ANSWER = self.STOP_STRINGS + ["\n\n", "."]
        
        predicted_answer = self.generate_text(
            answer_prompt,
            max_new_tokens=50,
            stop_strings=STOP_ANSWER
        )
        
        time_taken = time.time() - start_time
        
        # Clean up the predicted answer (remove markdown formatting)
        predicted_answer = predicted_answer.replace('**', '').replace('__', '').strip()
        
        # 3. Grade answer using compute_score (no need for extract_answer)
        if predicted_answer == row["answer"] or predicted_answer == f"{row['answer']}.":
            is_correct = True
        else:
            is_correct = self.grade_answer(predicted_answer, row["answer"])
        
        print(f"Question: {row['problem']}")
        print(f"Solution: {solution}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Ground Truth: {row['answer']}")
        print(f"Is Correct: {is_correct}")
        print("-" * 80)

        return {
            "unique_id": row['unique_id'],
            "problem": row['problem'],
            "solution": solution,
            "predicted_answer": predicted_answer,
            "answer": row['answer'],
            "is_correct": is_correct,
            "time_taken": time_taken
        }


def main(args):
    # Initialize the MATH500Processor with the provided configuration
    config = {
        "model": {
            "type": "transformers",
            "path": args.base_model,
            "cache_dir": args.cache_dir
        },
        "save_dir": args.save_dir,
        "shots": args.shots,
        "examples_path": args.examples_path,
        "error_log": args.error_log
    }
    
    math_processor = MATH500Processor(config)
    math_results = math_processor.run()

    # Compute accuracy
    accuracy = math_results['is_correct'].mean() * 100
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MATH-500 evaluation")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Base model identifier or path")
    parser.add_argument("--cache_dir", default='/data2/.shared_models',
                        help="Directory for storing base models")
    parser.add_argument("--save_dir", default="./evaluate/results/math500",
                        help="Directory for saving results")
    parser.add_argument("--shots", type=int, default=0,
                        help="Number of shots for few-shot evaluation")
    parser.add_argument("--examples_path", default=None,
                        help="Path to CSV file with few-shot examples")
    parser.add_argument("--error_log", default=None,
                        help="Path to error log file")
    parser.add_argument("--type", default="transformers",
                        choices=["transformers", "llama.cpp"],
                        help="Model type (transformers or llama.cpp)")

    args = parser.parse_args()
    main(args)