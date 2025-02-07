import pandas as pd
import json
import chardet
from huggingface_hub import login
from evaluator import PsychDepthEvaluator  

login(token='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')
# Detect file encoding
stories_path = "../data/stories/study_stories.csv"
model = "meta-llama/Llama-3.3-70B-Instruct"
with open(stories_path, "rb") as f:
    result = chardet.detect(f.read(100000))  # Read a portion of the file
    encoding = result["encoding"]
    print(f"Detected encoding: {encoding}")

# Load the dataset with detected encoding
stories_df = pd.read_csv(stories_path, encoding=encoding)

# Initialize evaluator
evaluator = PsychDepthEvaluator(
    model_id= model,
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
    print(story_id)
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
results_df.to_csv("../data/stories/evaluation_results_{model}.csv", index=False)

print("Results saved to ../data/stories/evaluation_results.csv")
