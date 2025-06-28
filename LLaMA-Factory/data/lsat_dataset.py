from datasets import load_dataset
import pandas as pd
import datasets
import json
import ast
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split

## Dataset preparation -- using reasoning traces from huggingface [cot-leaderboard/cot-eval-traces-2.0]
## The short-* data files in this directory are sampled from the complete set

for i in ['all', 'better']:
    path = f'lsat-ar-{i}-model-traces.csv'  # pre-filtered based on config_data['model] in [cot-leaderboard/cot-eval-traces-2.0]
    df = pd.read_csv(path)
    splits = {}
    splits['train'], temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    splits['val'], splits['test'] = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)
    for spl, df in splits.items():
        data = []
        for _, r in df.iterrows():
            data.append({
                'instruction': 'Answer the following question. Think step by step before giving the final answer.',
                'input': '\n\n'.join([r['passage'], r['question_options']]),
                'output': r['reasoning_trace']
            })
        with open(f'{spl}-lsat-ar-{i}-model-traces.json', 'w') as f:
            json.dump(data, f, indent=2)


## Dataset preparation -- only including ground truth, no reasoning trace provided

ds = load_dataset("tasksource/lsat-ar")

for split in ['validation', 'train', 'test']:
    ds_split = ds[split]
    ds_json = []
    for i in ds_split:
        try:
            choices = [int(j) for j in i['answers']]
            choices = 'The choices are ' + ', '.join([str(j) for j in choices])
        except:
            choices = ''
            for ind, choice in enumerate(i['answers']):
                choices += f'Choice {ind}: {choice}\n'
        ds_json.append({
            'instruction': "Answer the question with only the number of the correct choice. No explanation. Just one digit.",
            'input': '\n'.join([i['context'], i['question'], choices]),
            'output': str(i['label'])
        })
    with open(f"lsat-{split}.json", "w") as f:
        json.dump(ds_json, f, indent=2)
        