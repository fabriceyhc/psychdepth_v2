import argparse
import pandas as pd
import json

NUM_WORDS = 500

PROFILE = """\
You are a seasoned writer who has won several accolades for your emotionally rich stories. 
When you write, you delve deep into the human psyche, pulling from the reservoir of universal 
experiences that every reader, regardless of their background, can connect to.
Your writing is renowned for painting vivid emotional landscapes, making readers not just observe 
but truly feel the world of your characters.
Every piece you produce aims to draw readers in, encouraging them to reflect on their own lives 
and emotions.
Your stories are a complex tapestry of relationships, emotions, and conflicts, 
each more intricate than the last.
"""

SCORE_COLUMNS = [
    'authenticity_score',
    'emotion_provoking_score',
    'empathy_score',
    'engagement_score',
    'narrative_complexity_score'
]

def main(args):
    # Load and prepare data
    df = pd.read_csv(args.input)
    df['avg_score'] = df[SCORE_COLUMNS].mean(axis=1)

    # Apply filtering only for SFT
    if args.type == 'sft' and args.min_avg_pds_score > 0:
        df = df[df['avg_score'] > args.min_avg_pds_score]
    
    # Format data according to specified type
    formatted_data = []
    if args.type == 'sft':
        for _, row in df.iterrows():
            formatted_data.append({
                "conversations": [
                    {"from": "human", "value": f"{PROFILE}\n\nWrite a {NUM_WORDS}-word story on the following prompt:\n{row['premise']}\n\nOnly respond with the story."},
                    {"from": "gpt", "value": row['text']}
                ]
            })
    elif args.type == 'dpo':
        for premise, group_df in df.groupby('premise'):
            for i in range(len(group_df)):
                s1 = group_df.iloc[i]
                for j in range(i+1, len(group_df)):
                    s2 = group_df.iloc[j]
                    if abs(s1['avg_score'] - s2['avg_score']) >= args.threshold:
                        higher, lower = (s2, s1) if s1['avg_score'] < s2['avg_score'] else (s1, s2)
                        formatted_data.append({
                            "conversations": [
                                {"from": "human", "value": f"{PROFILE}\n\nWrite a {NUM_WORDS}-word story on the following prompt:\n{premise}\n\nOnly respond with the story."}
                            ],
                            "chosen": {"from": "gpt", "value": higher['text']},
                            "rejected": {"from": "gpt", "value": lower['text']}
                        })
    elif args.type == 'kto':
        # Use all data without filtering, label based on threshold
        for _, row in df.iterrows():
            formatted_data.append({
                "conversations": [
                    {"from": "human", "value": f"{PROFILE}\n\nWrite a {NUM_WORDS}-word story on the following prompt:\n{row['premise']}\n\nOnly respond with the story."},
                    {"from": "gpt", "value": row['text']}
                ],
                "label": row['avg_score'] >= args.min_avg_pds_score
            })
    
    # Save output
    with open(args.output, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"{len(formatted_data)} rows saved to {args.output}")

if __name__ == "__main__":

    # RUN: ./dataset/scripts/format.sh 

    parser = argparse.ArgumentParser(description='Convert dataset to ShareGPT format')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--type', type=str, required=True, choices=['sft', 'dpo', 'kto'], 
                       help='Type of format to generate: sft, dpo, or kto')
    parser.add_argument('--min_avg_pds_score', type=float, default=3.5,
                       help='Minimum average PDS score for filtering (SFT) or labeling (KTO)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Minimum score difference for DPO comparisons only')
    
    args = parser.parse_args()
    main(args)