import pandas as pd
import json

DIFF_THRESHOLD = 1  # Minimum score difference for comparison
df = pd.read_csv('data/scored_reddit_stories.csv')

# Compute total score as the sum of selected scores
scores = ['authenticity_score', 'emotion_provoking_score', 'empathy_score', 'engagement_score', 'narrative_complexity_score']
df['total_score'] = df[scores].sum(axis=1)

data = []
for premise, group_df in df.groupby('premise'):
    for idx in range(len(group_df)):
        s1 = group_df.iloc[idx]
        for j in range(idx + 1, len(group_df)):
            s2 = group_df.iloc[j]
            if abs(s1['total_score'] - s2['total_score']) >= DIFF_THRESHOLD:
                higher, lower = (s2, s1) if s1['total_score'] < s2['total_score'] else (s1, s2)
                data.append({'premise': premise, 'chosen': higher['story'], 'rejected': lower['story']})

kto_data = []
for item in data:
    # Create preference pairs for fine-tuning
    kto_data.append({
        "messages": [
            {"content": f"Write a story on the following prompt. Only respond with the story.\n{item['premise']}", "role": "user"},
            {"content": item['chosen'], "role": "assistant"}
        ],
        "label": True
    })
    kto_data.append({
        "messages": [
            {"content": f"Write a story on the following prompt. Only respond with the story.\n{item['premise']}", "role": "user"},
            {"content": item['rejected'], "role": "assistant"}
        ],
        "label": False
    })

# Save dataset for model training
with open("kto_reddit.json", "w") as f:
    json.dump(kto_data, f)