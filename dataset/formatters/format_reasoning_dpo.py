import json
import ast
from datasets import load_dataset

hf_dataset_path = ""     
subset = "" # manually check the subsets before choosing the appropriate one
output_json_path = ""
split = "default" # "train", "validation"
dataset = load_dataset(hf_dataset_path, subset)[split]

pairs = []

for item in dataset:
    traj = item["trajectory"] # dataset-specific
    for turn in traj:
        if turn["from"] == "assistant":
            turn["from"] = "gpt"
        else:
            turn["from"] = "human"

    pairs.append({
        "chosen": {
            "value": item["chosen"],
            "from": "gpt"
        },
        "rejected": {
            "value": item["rejected"],
            "from": "gpt"
        },
        "conversations": traj
    })


# 保存为 JSON 文件
with open(output_json_path, "w") as f:
    json.dump(pairs, f, indent=2, ensure_ascii=False)