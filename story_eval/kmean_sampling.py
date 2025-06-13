# import pandas as pd
# from sklearn.cluster import KMeans
# import numpy as np

# def pick_representative_demos(df, n_clusters=10, random_state=42):
#     # 1) Extract the 6 columns into a numpy array for clustering
#     score_cols = [
#         "authenticity_score",
#         "emotion_provoking_score",
#         "empathy_score",
#         "engagement_score",
#         "narrative_complexity_score",
#         "human_likeness_score"
#     ]
#     X = df[score_cols].values

#     # 2) Fit KMeans
#     kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
#     kmeans.fit(X)

#     # 3) For each cluster, find the row that is closest to the cluster center
#     #    We'll measure Euclidean distance from each point to its cluster center
#     #    and pick the minimal distance row per cluster.
#     labels = kmeans.labels_
#     centers = kmeans.cluster_centers_

#     best_indices = [-1]*n_clusters
#     best_distances = [float("inf")]*n_clusters

#     for i, row in enumerate(X):
#         cluster_id = labels[i]
#         center = centers[cluster_id]
#         dist = np.linalg.norm(row - center)  # Euclidean distance
#         if dist < best_distances[cluster_id]:
#             best_distances[cluster_id] = dist
#             best_indices[cluster_id] = i

#     # 4) best_indices now contains the row index in df for each cluster
#     chosen_rows = df.iloc[best_indices].copy()
#     return chosen_rows

# # Usage:
# df = pd.read_csv("./data/stories_w_human_annotations_multiscore_train.csv")

# import json

# def replace_demos_in_dspy_json(dspy_json_path, new_demos, out_json_path=None):
#     """
#     Replaces the 'demos' list in a DSPy JSON file with new_demos, 
#     then writes the updated JSON to disk (optionally to out_json_path).
#     """

#     # 1) Load the DSPy JSON
#     with open(dspy_json_path, "r", encoding="utf-8") as f:
#         dspy_program = json.load(f)

#     # 2) Replace the 'demos' list with your new demos
#     dspy_program['demos'] = new_demos

#     # 3) Save to a new file (or overwrite the existing one)
#     if out_json_path is None:
#         out_json_path = dspy_json_path  # overwrite
#     with open(out_json_path, "w", encoding="utf-8") as f:
#         json.dump(dspy_program, f, indent=2, ensure_ascii=False)

#     print(f"Replaced demos and wrote updated JSON to: {out_json_path}")

# if __name__ == "__main__":
#     # Suppose your new demos are 10 dictionaries, each with the same keys as the old demos.
#     # For example:
#     new_demos = []
#     # chosen = pick_representative_demos(df, n_clusters=5)
#     chosen = pick_representative_demos(df, n_clusters=5)
#     for _, row in chosen.iterrows():

#         demo_dict = {
#             "story": row["text"],
#             "authenticity_score": float(row["authenticity_score"]),
#             "emotion_provoking_score": float(row["emotion_provoking_score"]),
#             "empathy_score": float(row["empathy_score"]),
#             "engagement_score": float(row["engagement_score"]),
#             "narrative_complexity_score": float(row["narrative_complexity_score"]),
#             "human_likeness_score": float(row["human_likeness_score"])
#         }
#         new_demos.append(demo_dict)

#     # The path to your existing DSPy JSON
#     dsp_json_path = "./story_eval/dspy/multiscore/optimized_prompts/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/MIPROv2_Predict-PsychDepthAssessment_demos=5.json"

#     # Call the function to replace the demos
#     replace_demos_in_dspy_json(
#         dspy_json_path=dsp_json_path,
#         new_demos=new_demos,
#         out_json_path=dsp_json_path.replace("demos=5", "handpicked-demos=5")
#     )

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import json

def pick_representative_demos(df, n_clusters=10, random_state=42):
    # 1) Extract the 6 columns into a numpy array for clustering
    score_cols = [
        "authenticity_score",
        "emotion_provoking_score",
        "empathy_score",
        "engagement_score",
        "narrative_complexity_score",
        "human_likeness_score"
    ]
    X = df[score_cols].values

    # 2) Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)

    # 3) For each cluster, find the row that is closest to the cluster center
    #    We'll measure Euclidean distance from each point to its cluster center
    #    and pick the minimal distance row per cluster.
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    best_indices = [-1]*n_clusters
    best_distances = [float("inf")]*n_clusters

    for i, row in enumerate(X):
        cluster_id = labels[i]
        center = centers[cluster_id]
        dist = np.linalg.norm(row - center)  # Euclidean distance
        if dist < best_distances[cluster_id]:
            best_distances[cluster_id] = dist
            best_indices[cluster_id] = i

    # 4) best_indices now contains the row index in df for each cluster
    chosen_rows = df.iloc[best_indices].copy()
    return chosen_rows

# Usage:


def replace_demos_in_dspy_json(dspy_json_path, new_demos, out_json_path=None):
    """
    Replaces the 'demos' list in a DSPy JSON file with new_demos, 
    then writes the updated JSON to disk (optionally to out_json_path).
    """

    # 1) Load the DSPy JSON
    with open(dspy_json_path, "r", encoding="utf-8") as f:
        dspy_program = json.load(f)

    # 2) Replace the 'demos' list with your new demos
    dspy_program['demos'] = new_demos

    # 3) Save to a new file (or overwrite the existing one)
    if out_json_path is None:
        out_json_path = dspy_json_path  # overwrite
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(dspy_program, f, indent=2, ensure_ascii=False)

    print(f"Replaced demos and wrote updated JSON to: {out_json_path}")

if __name__ == "__main__":
    df = pd.read_csv("./data/multiscore/stories_w_human_annotations_multiscore_train.csv")
    # Suppose your new demos are 10 dictionaries, each with the same keys as the old demos.
    # For example
    # chosen = pick_representative_demos(df, n_clusters=5)
    n_demos = 10
    for participant in df['participant_id'].unique():
        new_demos = []
        subset = df[df['participant_id'] == participant]
        chosen = pick_representative_demos(subset, n_clusters=n_demos)
        for _, row in chosen.iterrows():
            demo_dict = {
                "story": row["text"],
                "authenticity_score": float(row["authenticity_score"]),
                "emotion_provoking_score": float(row["emotion_provoking_score"]),
                "empathy_score": float(row["empathy_score"]),
                "engagement_score": float(row["engagement_score"]),
                "narrative_complexity_score": float(row["narrative_complexity_score"]),
                "human_likeness_score": float(row["human_likeness_score"])
            }
            new_demos.append(demo_dict)

        dsp_json_path = f"./story_eval/dspy/multiscore/optimized_prompts/meta-llama/Llama-3.1-8B-Instruct/MIPROv2_Predict-PsychDepthAssessment_demos=10.json"
        replace_demos_in_dspy_json(
            dspy_json_path=dsp_json_path,
            new_demos=new_demos,
            out_json_path=dsp_json_path.replace(f"demos=10", f"kmean-demos={n_demos}_participant={participant}_persona")
        )