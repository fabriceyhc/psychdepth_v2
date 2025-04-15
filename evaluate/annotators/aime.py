import time
import pandas as pd
from typing import List, Dict
import guidance
from guidance import models, gen, select, user, system, assistant
from datasets import load_dataset

from evaluate.annotators._base import BaseDatasetProcessor

class AIMEProcessor(BaseDatasetProcessor):
    """Processor for MATH-500 dataset"""
    
    def load_dataset(self) -> pd.DataFrame:
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train").to_pandas()
        return dataset

    def _is_processed(self, row: Dict, existing: pd.DataFrame) -> bool:
        """Check if row already exists in saved results"""
        if 'id' in row and 'id' in existing.columns:
            return (existing["id"] == row["id"]).any()
        return False

    def grade_answer(self, predicted_answer, ground_truth) -> bool:
        """Grade the predicted answer"""
        return int(predicted_answer) == ground_truth

    def process_entry(self, row: Dict) -> Dict:

        # 1. Collect answer
        start_time = time.time()
        output = self.model + self.annotation_prompt(
            problem=row['problem']
        )
        time_taken = time.time() - start_time

        # 2. Grade answer
        is_correct = self.grade_answer(output['answer'], row["answer"])

        return {
            "id": row['id'],
            "problem": row['problem'],
            "solution": output['solution'],
            "predicted_answer": output['answer'],
            "answer": row['answer'],
            "is_correct": is_correct,
            "time_taken": time_taken
        }
    
    @guidance(dedent=True)
    def annotation_prompt(self, lm, problem: str):
        # with system():
        #     lm += f"You are an expert mathematician specializing in {subject} problems (difficulty: {level})"
        
        with user():
            lm += f"""Solve this problem step-by-step. Your answer should be numerical with one, two, or three digits. 
            
            Problem: {problem}
            """
            
        with assistant():
            lm += f"Step-by-step Solution:\n{gen(name='solution', stop=self.STOP_STRINGS, max_tokens=500)}"
            regex = r'\d+' # Separating this from below to avoid f-string syntax error
            lm += f"Final Answer:\n{gen(name='answer', regex=regex, max_tokens=3)}"
        return lm


if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0 python -m evaluate.annotators.aime

    config = {
        "model": {
            "type": "transformers",
            "path": "meta-llama/Llama-3.2-1B-Instruct",
            "cache_dir": "/data2/.shared_models"
        },
        "save_dir": "./evaluate/results/aime",
        "shots": 0  # Zero-shot for math problems
    }

    math_processor = AIMEProcessor(config)
    math_results = math_processor.run()
