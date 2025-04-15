import json
from datasets import load_dataset

hf_dataset_path = ""     
subset = "" # manually check the subsets before choosing the appropriate one
output_json_path = ""
split = "default" # "train", "validation"
dataset = load_dataset(hf_dataset_path, subset)[split]

sft_data = []

# dataset-specific data cleaning and restructuring 
prefix_template = "The following paragraphs each describe a set of %s objects arranged in a fixed order. The statements are logically consistent within each paragraph."
replace_template = "The following paragraph describes a set of %s objects arranged in a fixed order. The statements are logically consistent within the paragraph.\n"

for row in dataset:
    inputs = row["inputs"]
    choices = row["multiple_choice_targets"]
    correct_choice = row["targets"]
    if len(choices) != 1 or len(correct_choice) != 1:
        continue
    
    for num in ["three", "five", "seven"]:
        prefix = prefix_template % num
        if inputs.startswith(prefix):
            inputs = inputs[len(prefix):].strip()
            inputs = (replace_template % num) + inputs + \
                "\nPlease choose the single best answer from the options below.\n" + "\n".join(choices)
            break

    sft_data.append({
        "instruction": "You will be given a question and a list of options. Choose the single most appropriate option.",
        "input": inputs,
        "output": row["targets"][0]
    })

with open(output_json_path, "w") as f:
    json.dump(sft_data, f, indent=2)