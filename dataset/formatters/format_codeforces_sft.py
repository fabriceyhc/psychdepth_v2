from datasets import load_dataset, load_from_disk
from pathlib import Path
import os
import json
import pandas as pd
import re
from pathlib import Path
from datasets import load_from_disk, DatasetDict
import polars as pl

ds = load_dataset(
    "open-r1/codeforces-cots",
    name="solutions",     
    split="train",        
    #     streaming=True                     #o full download
)

ds = ds.shuffle(seed=42)       
train_valtest = ds.train_test_split(test_size=0.20, seed=42)  # 80 % train
temp          = train_valtest["test"]                        

val_test = temp.train_test_split(test_size=1/2, seed=42)     
train_ds = train_valtest["train"]
test_ds  = val_test["test"]           # 10 % test
val_ds   = val_test["train"]          # 10 % val

# Save to disk 
root = Path("/home/sarakhosravi/psychdepth_v2/data/open-r1/codeforces")
# root = Path("./data/open-r1/codeforces")
for split_name, split_ds in [("train", train_ds),
                             ("test", test_ds),
                             ("validation", val_ds)]:
    path = root / split_name
    os.makedirs(path, exist_ok=True)
    split_ds.save_to_disk(str(path))  # Hugging Face arrow format :contentReference[oaicite:1]{index=1}



root = Path("/home/sarakhosravi/psychdepth_v2/data/open-r1/codeforces")

# 1.  load every split
ds_train = load_from_disk(root / "train")
ds_val   = load_from_disk(root / "validation")
ds_test  = load_from_disk(root / "test")
train_set = ds_train.to_pandas()
# train_set = ds_train.to_polars()   #20x faster than pandas
# optional: bundle together for convenience
codeforces_cots = DatasetDict(
    train=ds_train,
    validation=ds_val,
    test=ds_test,
)

# print(train_set.columns)
"""
Index(['id', 'aliases', 'contest_id', 'contest_name', 'contest_type',
       'contest_start', 'contest_start_year', 'index', 'time_limit',
       'memory_limit', 'title', 'description', 'input_format', 'output_format',
       'examples', 'note', 'editorial', 'prompt', 'generation',
       'finish_reason', 'api_metadata', 'interaction_format', 'messages'],
      dtype='object')
"""
import json, re, itertools
from pathlib import Path
from datasets import load_from_disk, load_dataset
from tqdm import tqdm

# ---------- paths ----------
disk_root  = Path("/home/sarakhosravi/psychdepth_v2/data/open-r1/codeforces")  
out_root   = Path("/home/sarakhosravi/psychdepth_v2/LLaMA-Factory/data/codeforces/sft")
out_root.mkdir(parents=True, exist_ok=True)

# ---------- utils ----------
THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.S)
def strip_think(text: str) -> str:
    return THINK_RE.sub("", str(text))

def prompt_template(raw_prompt: str) -> str:
    """Optional: adjust template. The raw prompt already contains instructions."""
    return raw_prompt.strip()

# ---------- A) AI answers ----------------------------------------------------
train_ai = load_from_disk(disk_root / "train")     # HuggingFace Dataset
ai_records = []
for row in tqdm(train_ai, desc="AI split"):
    ai_records.append({
        "id":   row["id"],
        "conversations": [
            {"from": "human", "value": prompt_template(row["prompt"])},
            {"from": "gpt",   "value": strip_think(row["generation"])}
        ]
    })

(ai_path := out_root / "codeforces_ai_cots.json").write_text(
    json.dumps(ai_records, indent=2),
    encoding="utf-8"
)
print(f"✓ wrote {len(ai_records):,} AI examples → {ai_path}")

# ---------- B) Human editorials ---------------------------------------------
#  - download only once; it's ~25 k samples
human_ds = load_dataset(
    "open-r1/codeforces-cots",
    name="solutions_w_editorials",
    split="train",
    streaming=True
)

human_records = []
for row in tqdm(human_ds, desc="human split (stream)"):
    editorial = row.get("editorial", None)
    if not editorial:            # skip problems with no human write-up
        continue
    human_records.append({
        "id": row["id"],
        "conversations": [
            {"from": "human", "value": prompt_template(row["prompt"])},
            {"from": "gpt",   "value": editorial.strip()}
        ]
    })

(h_path := out_root / "codeforces_human_cots.json").write_text(
    json.dumps(human_records, indent=2),
    encoding="utf-8"
)

