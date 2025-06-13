import pandas as pd
import json

lim = ('', 'answer')
f = []
for i in range(50,400,50):
    path = f"temp0_{lim[0]}gsm8k_{lim[1]}_{i}.jsonl"
    with open(path, "r") as file:
        
        for line in file:
            f.append(json.loads(line))
            f[-1]['ckpt'] = i
f = pd.DataFrame(f)
f['predict'] = f['predict'].apply(str.strip)
f['label'] = f['label'].apply(str.strip)
stats = []
for ckpt, group in f.groupby('ckpt'):
    s = sum(group.predict == group.label)
    stats.append({'correct': s, 'ckpt': ckpt})
print(stats)
path = f"temp0_{lim[0]}_gsm8k_{lim[1]}_stats.json"
with open(path, "w") as f:
    json.dump(stats, f)