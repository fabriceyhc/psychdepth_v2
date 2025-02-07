import pandas as pd
import json
from evaluator import PsychDepthEvaluator  # Assuming the evaluator is in evaluator.py

# Load the dataset
stories_path = "../data/stories/study_stories.csv"
stories_df = pd.read_csv(stories_path)

# Initialize evaluator
evaluator = PsychDepthEvaluator(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    model_type="transformers",
    cache_dir="/data2/.shared_models/",
    device_map="auto",
    verbose=True
)

# Run evaluation on each story
results = []
for _, row in stories_df.iterrows():
    story_id = row["story_id"]
    story_text = row["text"]
    evaluation = evaluator.evaluate(story=story_text, personas=evaluator.personas, temperature=1.0)
    
    # Flatten results for saving
    for persona, scores in evaluation.items():
        results.append({
            "story_id": story_id,
            "persona": persona,
            **scores
        })

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("../data/stories/evaluation_results.csv", index=False)

print("Evaluation complete. Results saved to ../data/stories/evaluation_results.csv")
