from datasets import load_dataset,load_from_disk
import os
import json
import pandas as pd
import re
from pathlib import Path
from datasets import DatasetDict
#import polars as pl
from tqdm import tqdm

"""
Do this part once to download the dataset and save it locally
if want to redo it use these commands and delete the ./cache hugging face data folder:
conda deactivate
conda remove --name pds --all
conda create -n pds python=3.12
conda activate pds
pip install datasets
"""

# codeforces columns
"""
Index(['id', 'aliases', 'contest_id', 'contest_name', 'contest_type',
       'contest_start', 'contest_start_year', 'index', 'time_limit',
       'memory_limit', 'title', 'description', 'input_format', 'output_format',
       'examples', 'note', 'editorial', 'prompt', 'generation',
       'finish_reason', 'api_metadata', 'interaction_format', 'messages'],
      dtype='object')
"""

#ds = load_dataset("open-r1/codeforces-cots", name="solutions", split="train")
# ds = ds.shuffle(seed=42)       
# train_valtest = ds.train_test_split(test_size=0.20, seed=42)  # 80 % train
# temp          = train_valtest["test"]                        
# val_test = temp.train_test_split(test_size=1/2, seed=42)     
# train_ds = train_valtest["train"]
# test_ds  = val_test["test"]           # 10 % test
# val_ds   = val_test["train"]          # 10 % val
# root = Path("./data/open-r1/codeforces")
# for split_name, split_ds in [("train", train_ds),
#                              ("test", test_ds),
#                              ("validation", val_ds)]:
#     path = root / split_name
#     os.makedirs(path, exist_ok=True)
#     split_ds.save_to_disk(str(path))  # Hugging Face arrow format :contentReference[oaicite:1]{index=1}

from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from pathlib import Path
import json

# Paths
root = Path("data/open-r1/codeforces")
out_root = Path("./LLaMA-Factory/data/codeforces/sft")
out_root.mkdir(parents=True, exist_ok=True)

# Template cleaner
def prompt_template(raw_prompt: str) -> str:
    return raw_prompt.strip()

# --- Helper for deduplication by normalized JSON string ---
def to_canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

# ---------- A) AI answers ----------------------------------------------------
train_ai = load_from_disk(root / "train")   
ai_seen = set()
ai_records = []

for row in tqdm(train_ai, desc="AI split"):
    record = {
        "id": row["id"],
        "conversations": [
            {"from": "human", "value": prompt_template(row["prompt"])},
            {"from": "gpt",   "value": row["generation"]}
        ]
    }
    record_str = to_canonical_json(record)
    if record_str not in ai_seen:
        ai_seen.add(record_str)
        ai_records.append(record)

(ai_path := out_root / "codeforces_ai_cots.json").write_text(
    json.dumps(ai_records, indent=2),
    encoding="utf-8"
)
print(f"Wrote {len(ai_records):,} unique AI examples to: {ai_path}")

# ---------- B) Human editorials ---------------------------------------------
human_ds = load_dataset(
    "open-r1/codeforces-cots",
    name="solutions_w_editorials",
    split="train",
    streaming=True
)

human_seen = set()
human_records = []

for row in tqdm(human_ds, desc="Human split (stream)"):
    editorial = row.get("editorial", None)
    if not editorial:
        continue

    record = {
        "id": row["id"],
        "conversations": [
            {"from": "human", "value": prompt_template(row["prompt"])},
            {"from": "gpt",   "value": editorial.strip()}
        ]
    }
    record_str = to_canonical_json(record)
    if record_str not in human_seen:
        human_seen.add(record_str)
        human_records.append(record)

(h_path := out_root / "codeforces_human_cots.json").write_text(
    json.dumps(human_records, indent=2),
    encoding="utf-8"
)
print(f"Wrote {len(human_records):,} unique human examples to: {h_path}")
