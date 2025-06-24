import time
import pandas as pd
from typing import List, Dict
import guidance
from guidance import models, gen, select, user, system, assistant
from datasets import load_dataset
from .utils.math import compute_score
import argparse
import re

from psychdepth_v2.evaluate.annotators._base_guidance import BaseDatasetProcessor

class MATH500Processor(BaseDatasetProcessor):
    """Processor for MATH-500 dataset"""
    
    def load_dataset(self) -> pd.DataFrame:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test").to_pandas()[:100]
        return dataset

    def _is_processed(self, row: Dict, existing: pd.DataFrame) -> bool:
        """Check if row already exists in saved results"""
        if 'id' in row and 'id' in existing.columns:
            return (existing["id"] == row["id"]).any()
        return False

    def grade_answer(self, predicted_answer, ground_truth) -> bool:
        """Grade the predicted answer"""
        # TODO: Implement Latex equivalence checking
        # return int(predicted_answer) == ground_truth
        return compute_score(predicted_answer, ground_truth)

    def process_entry(self, row: Dict) -> Dict:

        # 1. Collect answer
        start_time = time.time()
        output = self.model + self.annotation_prompt(
            problem=row['problem']
        )
        time_taken = time.time() - start_time
        # Extract answer after "Final Answer" text
        raw_answer = output['answer']
        
        # Handle different formats using split/partition
        answer = raw_answer.partition('Final Answer:')[-1]  # Get text after last occurrence
        answer = answer.split('\n')[0].strip()  # Take first line/segment
        answer = answer.replace('**', '').replace('__', '')  # Remove markdown formatting

        print(answer)
        
        output.set('answer', answer)
        # ['answer'] = answer

        # 2. Grade answer
        is_correct = self.grade_answer(output['answer'], row["answer"])

        return {
            "unique_id": row['unique_id'],
            "problem": row['problem'],
            "solution": output['solution'],
            "predicted_answer": output['answer'],
            "answer": row['answer'],
            "is_correct": is_correct,
            "time_taken": time_taken
        }
    
    @guidance(dedent=True)
    def annotation_prompt(self, lm, problem: str):
        with system():
            lm += f"detailed thinking on\n"
        with user():
            lm += f"""Please reason step by step, and put your final answer within \\boxed{{}}.
            
            Problem: {problem}
            """
            
        # self.STOP_STRINGS.append("Final Answer:")
        STOP_SOLUTION = self.STOP_STRINGS + ["Final Answer:"]  
        with assistant():
            # lm += f"Step-by-step Solution:\n{gen(name='solution', stop=self.STOP_STRINGS, max_tokens=1024)}\n\n"
            # lm += f"Only output the Final Answer here. Put Final answer inside \\boxed{{}}. Do not include any further reasoning. Final Answer:\n{gen(name='answer', max_tokens=50, stop=self.STOP_STRINGS)}"
            lm += (
                "Step-by-step Solution:\n"
                # 1️⃣ let the model keep talking until it types "Final Answer:"
                f"{gen(name='solution', stop=STOP_SOLUTION , max_tokens=4096)}"
                # 2️⃣ now we are *after* the stop string, so the cursor is right
                #     after “Final Answer:”.  Time to collect *only* the boxed answer.
                "\nPut your final answer within \\boxed{{}}. Final Answer:\n"
                f"\\boxed{gen(name='answer', stop=self.STOP_STRINGS, max_tokens=50)}"
        )
        return lm

def main(args):
    # Initialize the MATH500Processor with the provided configuration
    config = {
        "model": {
            "type": args.type,
            "path": args.base_model,
            "cache_dir": args.cache_dir
        },
        "save_dir": args.save_dir,
        "shots": args.shots
    }
    
    math_processor = MATH500Processor(config)
    math_results = math_processor.run()

    # Compute accuracy
    accuracy = math_results['is_correct'].mean() * 100
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run AIME evaluation")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Base model identifier or path")
    parser.add_argument("--cache_dir", default='/data2/.shared_models',
                        help="Directory for storing base models")
    parser.add_argument("--type", default="transformers",
                        choices=["transformers", "llama.cpp"],
                        help="Model type (transformers or llama.cpp)")
    parser.add_argument("--save_dir", default="./evaluate/results/aime_1983_2024",
                        help="Directory for saving results")
    parser.add_argument("--shots", type=int, default=0,
                        help="Number of shots for zero-shot evaluation")

    # CUDA_VISIBLE_DEVICES=3 python -m evaluate.annotators.math-500

    args = parser.parse_args()
    main(args)

# class MATH500Processor(BaseDatasetProcessor):
#     """Processor for MATH-500 dataset"""
    
#     def load_dataset(self) -> pd.DataFrame:
#         dataset = load_dataset("HuggingFaceH4/MATH-500", split="test").to_pandas()[:100]
#         return dataset

#     def _is_processed(self, row: Dict, existing: pd.DataFrame) -> bool:
#         """Check if row already exists in saved results"""
#         if 'unique_id' in row and 'unique_id' in existing.columns:
#             return (existing["unique_id"] == row["unique_id"]).any()
#         return False

#     def grade_answer(self, predicted_answer, ground_truth) -> bool:
#         """Grade the predicted answer"""
#         return compute_score(predicted_answer, ground_truth)

#     def create_prompt(self, problem: str) -> str:
#         """Create a prompt for the model without using guidance"""
#         # Base prompt with no examples
#         if not self.examples or self.shots == 0:
#             prompt = f"""Please reason step by step, and put your final answer within \\boxed{{}}.

# Problem: {problem}"""
#             return prompt
        
#         # Add few-shot examples if available
#         prompt = "Please solve the following math problems step by step, and put your final answer within \\boxed{}.\n\n"
        
#         # Add examples based on the number of shots
#         for i in range(min(self.shots, len(self.examples))):
#             example = self.examples.iloc[i]
#             prompt += f"Problem: {example['problem']}\n\n"
#             prompt += f"Step-by-step Solution:\n{example['solution']}\n\n"
#             prompt += f"Final Answer: {example['answer']}\n\n"
        
#         # Add the current problem
#         prompt += f"Problem: {problem}"
#         return prompt

#     def process_entry(self, row: Dict) -> Dict:
#         # 1. Create the prompt
#         prompt = self.create_prompt(row['problem'])
        
#         # 2. Generate the solution
#         start_time = time.time()
#         solution = self.generate_text(prompt, max_new_tokens=1024, stop_strings=self.STOP_STRINGS)
#         time_taken = time.time() - start_time
        
#         # 3. Extract answer (look for Final Answer or \boxed{})
#         answer = self.extract_answer(solution)
        
#         # 4. Grade answer
#         is_correct = self.grade_answer(answer, row["answer"])
#         print(f"Questoin: {row['problem']}")
#         print(f"Solution: {solution}")
#         print(f"Predicted Answer: {answer}")
#         print(f"Ground Truth: {row['answer']}")
#         print(f"Is Correct: {is_correct}")

#         return {
#             "unique_id": row['unique_id'],
#             "problem": row['problem'],
#             "solution": solution,
#             "predicted_answer": answer,
#             "answer": row['answer'],
#             "is_correct": is_correct,
#             "time_taken": time_taken
#         }
    
#     def extract_answer(self, solution: str) -> str:
#         """Extract the final answer from the solution text"""
#         # Try to find "Final Answer:" first
#         if "Final Answer:" in solution:
#             answer = solution.partition("Final Answer:")[-1]
#             answer = answer.split('\n')[0].strip()
#         # Then try to find \boxed{} content
#         elif "\\boxed{" in solution:
#             boxed_matches = re.findall(r"\\boxed{(.*?)}", solution)
#             if boxed_matches:
#                 answer = boxed_matches[-1]  # Take the last boxed content
#             else:
#                 answer = ""
#         else:
#             # If no clear indication, take the last line as the answer
#             answer = solution.strip().split('\n')[-1]
            
#         # Clean up formatting
#         answer = answer.replace('**', '').replace('__', '')
#         return answer


# def main(args):
#     # Initialize the MATH500Processor with the provided configuration
#     config = {
#         "model": {
#             "path": args.base_model,
#             "cache_dir": args.cache_dir
#         },
#         "save_dir": args.save_dir,
#         "shots": args.shots,
#         "examples_path": args.examples_path,
#         "error_log": args.error_log
#     }
    
#     math_processor = MATH500Processor(config)
#     math_results = math_processor.run()

#     # Compute accuracy
#     accuracy = math_results['is_correct'].mean() * 100
#     print(f"Accuracy: {accuracy:.2f}%")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run MATH-500 evaluation")
#     parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct",
#                         help="Base model identifier or path")
#     parser.add_argument("--cache_dir", default='/data2/.shared_models',
#                         help="Directory for storing base models")
#     parser.add_argument("--save_dir", default="./evaluate/results/math500",
#                         help="Directory for saving results")
#     parser.add_argument("--shots", type=int, default=0,
#                         help="Number of shots for few-shot evaluation")
#     parser.add_argument("--examples_path", default=None,
#                         help="Path to CSV file with few-shot examples")
#     parser.add_argument("--error_log", default=None,
#                         help="Path to error log file")
#     parser.add_argument("--type", default="transformers",
#                         choices=["transformers", "llama.cpp"],
#                         help="Model type (transformers or llama.cpp)")

#     args = parser.parse_args()
#     main(args)