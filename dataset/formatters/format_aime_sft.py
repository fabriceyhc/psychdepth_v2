# Run: python -m dataset.formatters.format_aime_sft

import json
import pandas as pd
from datasets import load_from_disk
import re

hf_dataset_path = "di-zhang-fdu/AIME_1983_2024"     
subset = "" # manually check the subsets before choosing the appropriate one
train_set = load_from_disk("./data/open-r1/OpenR1-Math-220k/amc_aime")['train'].to_pandas()
# train_file = './dataset/data/AIME_Dataset_1983_2024_train.csv'
# train_set = pd.read_csv(train_file)

def clean_solution(solution_text):
    """
    Clean the solution text by:
    1. Removing name tags that appear in the middle or end of text
    2. Removing hyperlinks (http and https)
    
    Args:
        solution_text (str): The raw solution text containing name tags and/or hyperlinks
        
    Returns:
        str: Cleaned solution text with name tags and hyperlinks removed
    """
    # Convert to string if it's not already
    if solution_text is None:
        return ""
    solution_text = str(solution_text)
    
    # Pattern 1: Name tags that are on their own line (starting with \n~ and ending with \n)
    clean_text = re.sub(r'~[^\n]+\n', '\n', solution_text)
    
    # Pattern 2: Name tags at the end of text (starting with \n~)
    clean_text = re.sub(r'~[^\n]*$', '', clean_text)
    
    # Pattern 3: Remove hyperlinks - match http/https URLs with various possible endings
    # This handles standard URLs and complex ones with parentheses
    clean_text = re.sub(r'https?://[^\s()<>]+(?:\([^\s()<>]*\)|[^\s`!()\[\]{};:\'".,<>?«»""''])*', '', clean_text)
    
    # Special case for complex URLs with embedded http/https like in the example
    clean_text = re.sub(r'\([^()]*https?://[^()]*\)', '', clean_text)
    
    # Clean up any consecutive spaces that might be left after removing links
    clean_text = re.sub(r' +', ' ', clean_text)
    
    return clean_text

output_json_path = "./LLaMA-Factory/data/aime/sft/amc_aime_sft_human_solution.json"
sft_data = []
for index, row in train_set.iterrows():
    question = row["problem"]
    answer = clean_solution(row["solution"])

    formatted_input = (
        "Please reason step by step, and put your final answer within \\boxed{}:\n"
        f"{question}\n"
    )

    sft_data.append({
        "conversations": [
            {"from": "human", "value": formatted_input},
            {"from": "gpt", "value": str(answer)}
        ]
    })

with open(output_json_path, "w") as f:
    json.dump(sft_data, f, indent=2)

output_json_path = "./LLaMA-Factory/data/aime/sft/amc_aime_sft_ai_cot_solution.json"
sft_data = []
for index, row in train_set.iterrows():
    question = row["problem"]
    answer = row["messages"][-1]["content"]

    formatted_input = (
        "Please reason step by step, and put your final answer within \\boxed{}:\n"
        f"{question}\n"
    )

    sft_data.append({
        "conversations": [
            {"from": "human", "value": formatted_input},
            {"from": "gpt", "value": str(answer)}
        ]
    })

with open(output_json_path, "w") as f:
    json.dump(sft_data, f, indent=2)