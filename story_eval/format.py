import argparse
import pandas as pd
import json
import os

SCORE_COLUMNS = [
    'authenticity_score',
    'emotion_provoking_score',
    'empathy_score',
    'engagement_score',
    'narrative_complexity_score',
    'human_likeness_score'
]

EVALUATION_INSTRUCTION = """\
Imagine that you are a renowned literary critic, and your task is to evaluate the psychological depth of a given story. Your ratings will determine the story's eligibility for a prestigious literary award, and the author's career hangs in the balance.

Carefully review the given components of psychological depth: authenticity, emotion provoking, empathy, engagement, and narrative complexity. Be sure to understand each concept and the questions that characterize them.

Read the provided story, paying special attention to components of psychological depth. Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is average, and 5 is greatly above average (should be rare to provide this score).

Lastly, estimate the likelihood that the story was authored by a human or an LLM. Think about what human or LLM writing characteristics may be. Assign a score from 1 to 5, where 1 means very likely LLM written and 5 means very likely human written.

Your ratings and assessment will have a significant impact on the author's career, so be thorough and thoughtful in your evaluation.

###Description of Psychological Depth Components:  
    We define psychological depth in terms of the following concepts, each illustrated by several questions: 
    - Authenticity 
        - Does the writing feel true to real human experiences? 
        - Does it represent psychological processes in a way that feels authentic and believable? 
    - Emotion Provoking 
        - How well does the writing depict emotional experiences? 
        - Does it explore the nuances of the characters' emotional states, rather than just describing them in simple terms? 
        - Can the writing show rather than tell a wide variety of emotions? 
        - Do the emotions that are shown in the text make sense in the context of the story? 
    - Empathy 
        - Do you feel like you were able to empathize with the characters and situations in the text? 
        - Do you feel that the text led you to introspection, or to new insights about yourself or the world?\" 
    - Engagement
        - Does the text engage you on an emotional and psychological level? 
        - Do you feel the need to keep reading as you read the text? 
    - Narrative Complexity 
        - Do the characters in the story have multifaceted personalities? Are they developed beyond stereotypes or tropes? Do they exhibit internal conflicts?
        - Does the writing explore the complexities of relationships between characters?
        - Does it delve into the intricacies of conflicts and their partial or complete resolutions?
        
    Please provide your ratings and assessment in the format below:
    
    Authenticity Score: [insert score]
    Emotion Provoking Score: [insert score]
    Empathy Score: [insert score]
    Engagement Score: [insert score]
    Narrative Complexity Score: [insert score]
    Human Likeness Score: [insert score]
"""

def score_mse(human_score, llm_score):
    squared_error = 0
    for score in SCORE_COLUMNS:
        squared_error += (human_score[score] - llm_score[score])**2
    
    mean_squared_error = squared_error / len(SCORE_COLUMNS)
    return mean_squared_error


def main(args):
    # Load and prepare data
    df_best_llm = pd.read_csv(args.input_best_llm)
    df_worst_llm = pd.read_csv(args.input_worst_llm)
    df_stories = pd.read_csv("./data/stories/study_stories.csv")
    df_best_llm = df_best_llm.merge(df_stories, on="story_id")
    df_worst_llm = df_worst_llm.merge(df_stories, on="story_id")

    for index, value in df_best_llm["story_id"].items():
        diff = 0
        for metric in SCORE_COLUMNS:
            diff += (df_best_llm.loc[index, metric] - df_worst_llm.loc[index, metric])**2
        mse = diff/len(SCORE_COLUMNS)
        if mse <= args.threshold:
            df_best_llm = df_best_llm.drop(index=index)
            df_worst_llm = df_worst_llm.drop(index=index)
    
    best_save_path = args.output + "/" + "stories_w_" + "_".join(args.input_best_llm.split("/")[-1].split("_")[:2])+"_annotations_multiscore_train.csv"
    os.makedirs(os.path.dirname(best_save_path), exist_ok=True)
    df_best_llm.to_csv(best_save_path, index=False)
    worst_save_path = args.output + "/" + "stories_w_" + "_".join(args.input_worst_llm.split("/")[-1].split("_")[:2])+"_annotations_multiscore_train.csv"
    os.makedirs(os.path.dirname(worst_save_path), exist_ok=True)
    df_worst_llm.to_csv(worst_save_path, index=False)

if __name__ == "__main__":
    # Run sh ./story_eval/scripts/format.sh 
    
    parser = argparse.ArgumentParser(description='Splitting the dataset for post training')
    parser.add_argument('--input_best_llm', type=str, required=True, help='Input CSV file path for the llm annotated dataset')
    parser.add_argument('--input_worst_llm', type=str, required=True, help='Input CSV file path for the llm annotated dataset')
    parser.add_argument('--output', type=str, required=True, help='Output folder of training dataset')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Minimum score difference to be consider a valid pair')
    args = parser.parse_args()
    main(args)