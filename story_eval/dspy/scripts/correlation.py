import pandas as pd
from scipy.stats import spearmanr
import os
import json

if __name__ == "__main__":
    numeric_columns = [
        "authenticity_score", "empathy_score", "engagement_score", 
        "emotion_provoking_score", "narrative_complexity_score", "human_likeness_score"
    ]
    # load the single score testset
    single_score_df = pd.read_csv("./data/stories_w_human_annotations_singlescore_test.csv")
    single_score_df = single_score_df[["story_id", "participant_id", "pds_component", "score"]]
    single_score_df = single_score_df.pivot_table(
        index=["story_id", "participant_id"],
        columns="pds_component",
        values="score",
        aggfunc="first"
    )
    single_score_df.columns.name = None  # Remove the column grouping name
    single_score_df.columns = [str(col) for col in single_score_df.columns]
    single_score_df.rename(columns={"authenticity": "authenticity_score", "emotion_provoking": "emotion_provoking_score",
        "empathy": "empathy_score", "engagement": "engagement_score", 
        "human_likeness": "human_likeness_score", "narrative_complexity": "narrative_complexity_score"}, inplace=True)
    single_score_df[numeric_columns] = single_score_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    # Aggregate human scores by story_id (taking the mean across participants)
    single_score_df = single_score_df.groupby("story_id")[numeric_columns].mean().reset_index()
    # load the multi score testset
    multiscore_df = pd.read_csv("./data/stories_w_human_annotations_multiscore_test.csv")
    multiscore_df[numeric_columns] = multiscore_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    # Aggregate human scores by story_id (taking the mean across participants)
    multiscore_df = multiscore_df.groupby("story_id")[numeric_columns].mean().reset_index()

    # Load annotations
    all_summary = {}
    for file in os.listdir("./story_eval/dspy/dspy_annotations/"):
        # Compute correlations for singlescore annotations
        if "DepthS" in file or "DepthSE" in file or "DepthES" in file:
            print(f"result for {file}")
            rating_df = pd.read_csv("./story_eval/dspy/dspy_annotations/"+file)
            rating_df = rating_df.drop_duplicates(subset=["story_id", "psychological_depth_component"])
            rating_df = rating_df.pivot(
                index=["story_id"],
                columns="psychological_depth_component",
                values="score"
                ).reset_index()
            rating_df.columns.name = None  # Remove the column grouping name
            rating_df.columns = [str(col) for col in rating_df.columns]
            rating_df.rename(columns={"authenticity": "authenticity_score", "emotion_provoking": "emotion_provoking_score",
                "empathy": "empathy_score", "engagement": "engagement_score", 
                "human_likeness": "human_likeness_score", "narrative_complexity": "narrative_complexity_score"}, inplace=True)

            # Merge AI and human annotations on story_id
            merged_df = pd.merge(rating_df, single_score_df, on="story_id", suffixes=("_ai", "_human"))
            # Compute Spearman correlation for each score type
            correlations = {}
            for score in numeric_columns:
                r, p = spearmanr(merged_df[f"{score}_ai"], merged_df[f"{score}_human"])
                correlations[score] = {"correlation": r, "p_value": p}
            total = 0
            output = {}
            # Print correlation results
            for score, result in correlations.items():
                total += result['correlation'] if score != "human_likeness_score" else 0
                print(f"{score}: r={result['correlation']:.3f}, p={result['p_value']:.3f}")
                output[score] = result
            output["average"] = total / 5   
            # print(output)
            all_summary[file] = output
        
        # Compute annotations for multiscore annotations
        else:
            print(f"result for {file}")
            rating_df = pd.read_csv("./story_eval/dspy/dspy_annotations/"+file)
            # Merge AI and human annotations on story_id
            merged_df = pd.merge(rating_df, multiscore_df, on="story_id", suffixes=("_ai", "_human"))
            # Compute Spearman correlation for each score type
            correlations = {}
            for score in numeric_columns:
                r, p = spearmanr(merged_df[f"{score}_ai"], merged_df[f"{score}_human"])
                correlations[score] = {"correlation": r, "p_value": p}
            total = 0
            output = {}
            # Print correlation results
            for score, result in correlations.items():
                total += result['correlation'] if score != "human_likeness_score" else 0
                print(f"{score}: r={result['correlation']:.3f}, p={result['p_value']:.3f}")
                output[score] = result
            output["average"] = total / 5   
            # print(output)
            all_summary[file] = output
    with open("./story_eval/dspy/dspy_correlation_summary.json", "w") as f:
        json.dump(all_summary, f, indent=4)