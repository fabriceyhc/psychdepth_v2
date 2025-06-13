import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt

data = []

metrics = ["empathy", "emotion_provoking", 'narrative_complexity',"engagement",  "human_likeness", "authenticity"]
metrics = [i + '_score' for i in metrics]
models = ['llama_base', 'gsm8k-kto', 'openr1-dpo', 'ultrainteract-dpo', 'shuffled']
models = ['baseline', 'ground_truth_answer', 'shuffled']
for i in models:
    # df = pd.read_csv(f"guidance_evaled/guidance_evaled_{i}.csv")
    df = pd.read_csv(f"dataset/data/gsm8k/stories_{i}.csv")
    for col in metrics:
        data.append({
            "model": i,
            "score": col,
            "value": df[col].mean()
        })

table_data = []

for metric in metrics:
    row = [metric.replace('_score', '').title()]
    for model in models:
        value = next((d['value'] for d in data if d['model'] == model and d['score'] == metric), None)
        row.append(f"{value:.4f}" if value is not None else "N/A")
    table_data.append(row)

headers = ["Metric"] + [m.upper() for m in models]
print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))


df_heatmap = pd.DataFrame(table_data, columns=["Metric"] + [m.upper() for m in models])
df_heatmap.set_index("Metric", inplace=True)

df_heatmap = df_heatmap.map(lambda x: float(x) if x != "N/A" else None)

plt.figure(figsize=(10, 6))
sns.heatmap(df_heatmap, annot=True, cmap="YlOrRd", fmt=".4f", linewidths=0.5, linecolor='gray', cbar=True)

plt.title("Average Scores by Metric and Model", fontsize=14)
plt.ylabel("Metric")
plt.xlabel("Model")
plt.tight_layout()
# plt.show()
plt.savefig("comparison.png")

"""
gsm8k_answer: dominant
gsm8k_shuffled: dominant
math500_easy_answer: dominant
math500_easy_solution: dominant
math500_hard_solution: dominant
ultrainteract: dominant
gsm8k_kto: dominant

math_openr1_kto: lower empathy, authenticity
bigbench_deduction: lower authenticity
bigbench_narrative: lower authenticity
code_openr1: lower empathy
gsm8k_correct_output: lower empathy, narrative_complexity, engagement, human_likeness, authenticity
gsm8k_solution: lower narrative_complexity, engagement, human_likeness, authenticity
math500_hard_answer: lower empathy
"""