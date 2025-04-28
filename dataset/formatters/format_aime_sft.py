import json
import pandas as pd

hf_dataset_path = "di-zhang-fdu/AIME_1983_2024"     
subset = "" # manually check the subsets before choosing the appropriate one
output_json_path = "./LLaMA-Factory/data/pdsv2/sft/AIME_1983_2024_sft.json"
train_file = './dataset/data/AIME_Dataset_1983_2024_train.csv'
train_set = pd.read_csv(train_file)
sft_data = []


for index, row in train_set.iterrows():
    question = row["Question"]
    answer = row["Answer"]

    formatted_input = (
        "Solve the following problem and provide an answer:\n"
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