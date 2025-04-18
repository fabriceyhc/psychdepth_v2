import json
from datasets import load_dataset

hf_dataset_path = "allenai/ai2_arc"     
subset = "ARC-Challenge" # manually check the subsets before choosing the appropriate one
output_json_path = "../../LLaMA-Factory/data/arc/sft/arc_challenge_train.json"
split = "train" # "train", "validation"
dataset = load_dataset(hf_dataset_path, subset)[split]

sft_data = []

# dataset-specific data cleaning and restructuring 
prefix_template = "The following paragraphs each describe a set of %s objects arranged in a fixed order. The statements are logically consistent within each paragraph."
replace_template = "The following paragraph describes a set of %s objects arranged in a fixed order. The statements are logically consistent within the paragraph.\n"

for row in dataset:
    instr = "Answer the following question. Respond only with the chosen label."
    question = row['question']
    choices_raw = row['choices']
    answer = row['answerKey']
    assert len(choices_raw['text']) == len(choices_raw['label']) # avoid unexpected corner cases in messy datasets
    assert len(answer) == 1
    
    choices = ''
    for text, label in zip(choices_raw['text'], choices_raw['label']):
        choices += f'\n{label}: {text}'

    sft_data.append({
        "messages": [
            {"from": "human", "value": '\n'.join([instr, question, choices])},
            {"from": "gpt", "value": answer}
        ]
    }) # sharegpt format

with open(output_json_path, "w") as f:
    json.dump(sft_data, f, indent=2)