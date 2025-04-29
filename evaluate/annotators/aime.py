import time
import re
import pandas as pd
from typing import Dict
import guidance
from guidance import models, gen, select, user, system, assistant
from datasets import load_dataset, load_from_disk
from .utils.math import compute_score
import argparse

from evaluate.annotators._base import BaseDatasetProcessor

class AIMEProcessor(BaseDatasetProcessor):
    """Processor for AIME dataset"""
    
    def load_dataset(self) -> pd.DataFrame:
        # dataset = load_dataset("HuggingFaceH4/aime_2024", split="train").to_pandas()
        # dataset = pd.read_csv("./dataset/data/AIME_Dataset_1983_2024_test.csv")[:100]
        dataset = load_from_disk("./data/open-r1/OpenR1-Math-220k/amc_aime")["test"].to_pandas()
        return dataset

    def _is_processed(self, row: Dict, existing: pd.DataFrame) -> bool:
        """Check if row already exists in saved results"""
        if 'id' in row and 'id' in existing.columns:
            return (existing["id"] == row["id"]).any()
        return False

    # def grade_answer(self, predicted_answer, ground_truth) -> bool:
    #     """Grade the predicted answer"""
    #     return int(predicted_answer) == int(ground_truth)
    def grade_answer(self, predicted_answer, ground_truth) -> bool:
        """Grade the predicted answer"""
        return compute_score(predicted_answer, ground_truth)

    def process_entry(self, row: Dict) -> Dict:

        # # 1. Collect answer
        # start_time = time.time()
        # output = self.model + self.annotation_prompt(
        #     problem=row['problem']
        # )
        # time_taken = time.time() - start_time
        
        # # 2. Extract answer using regex
        # match = re.search(r'\d+', output['answer'])
        # extracted_answer = match.group(0) if match else ""

        # # 3. Grade answer using the extracted answer
        # is_correct = self.grade_answer(extracted_answer, row['answer'])

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
            "id": row['uuid'],
            "problem": row['problem'],
            "solution": output['solution'],
            "predicted_answer": output['answer'],
            "answer": row['answer'],
            "is_correct": is_correct,
            "time_taken": time_taken
        }
    
    @guidance(dedent=True)
    def annotation_prompt(self, lm, problem: str):
        with user():
            lm += f"""Please reason step by step, and put your final answer within \\boxed{{}}. 
            
            Problem: {problem}
            """
            
        with assistant():
            lm += f"Step-by-step Solution:\n{gen(name='solution', stop=self.STOP_STRINGS, max_tokens=1000)}"
            # lm += f"Final Answer:\n{gen(name='answer', stop=self.STOP_STRINGS, max_tokens=50)}"
            lm += f"Final Answer:\n{gen(name='answer', max_tokens=50, stop=self.STOP_STRINGS)}"
        return lm
    

def main(args):
    # Initialize the AIMEProcessor with the provided configuration
    config = {
        "model": {
            "type": args.type,
            "path": args.base_model,
            "cache_dir": args.cache_dir
        },
        "save_dir": args.save_dir,
        "shots": args.shots
    }
    
    math_processor = AIMEProcessor(config)
    math_results = math_processor.run()

    # Compute accuracy
    accuracy = math_results['is_correct'].mean() * 100
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=3 python -m evaluate.annotators.aime

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

    args = parser.parse_args()
    main(args)