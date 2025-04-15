import json
import ast
from datasets import load_dataset

hf_dataset_path = ""     
subset = "" # manually check the subsets before choosing the appropriate one
output_json_path = ""
split = "default" # "train", "validation"
dataset = load_dataset(hf_dataset_path, subset)[split]

kto_data = []

# dataset-specific data cleaning and restructuring 
prefix_template = "The following paragraphs each describe a set of %s objects arranged in a fixed order. The statements are logically consistent within each paragraph."
replace_template = "The following paragraph describes a set of %s objects arranged in a fixed order. The statements are logically consistent within the paragraph.\n"

for row in dataset:
    inputs = row['inputs']
    for num in ["three", "five", "seven"]:
        prefix = prefix_template % num
        if inputs.startswith(prefix):
            inputs = inputs[len(prefix):].strip()
            inputs = (replace_template % num) + inputs
            break

    steps = ast.literal_eval(row['model_output_steps'])
    output = '\n'.join(steps)
    label = row['model_output_solution_correctness'] == 'correct'
    kto_data.append({
        "messages": [
            {"role": "human", "content": inputs},
            {"role": "gpt", "content": output}
        ],
        "label": label
    })

# 输出文件名可根据需要手工修改
with open(output_json_path, "w") as f:
    json.dump(kto_data, f, indent=2)