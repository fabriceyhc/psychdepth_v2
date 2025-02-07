import pandas as pd
from scipy.stats import pearsonr

model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# Load AI-generated annotations
ai_df = pd.read_csv(f'../data/stories/evaluation_results_{model}.csv')

# Keep only the 'Average across personas' rows
ai_df = ai_df[ai_df["persona"] == "Average across personas"].drop(columns=["persona", "persona_id", "time_taken"])

# Load human annotations
human_df = pd.read_csv("../data/human_annotations/human_annotations.csv")

# Ensure numeric columns are properly converted
numeric_columns = [
    "authenticity_score", "empathy_score", "engagement_score", 
    "emotion_provoking_score", "narrative_complexity_score", "human_likeness_score"
]
human_df[numeric_columns] = human_df[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Aggregate human scores by story_id (taking the mean across participants)
human_df = human_df.groupby("story_id")[numeric_columns].mean().reset_index()

# Merge AI and human annotations on story_id
merged_df = pd.merge(ai_df, human_df, on="story_id", suffixes=("_ai", "_human"))

# Compute Pearson correlation for each score type
correlations = {}
for score in numeric_columns:
    r, p = pearsonr(merged_df[f"{score}_ai"], merged_df[f"{score}_human"])
    correlations[score] = {"correlation": r, "p_value": p}

# Print correlation results
for score, result in correlations.items():
    print(f"{score}: r={result['correlation']:.3f}, p={result['p_value']:.3f}")
