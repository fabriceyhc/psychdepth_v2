import os
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
import argparse
import traceback
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables first
load_dotenv(find_dotenv())  # Load OPENAI_API_KEY from ./.env

class OpenAIGenerator:
    """Direct OpenAI generator without guidance"""
    
    def __init__(
        self,
        model_id="gpt-3.5-turbo",
        max_input_len=4096,
        verbose=False
    ):
        """
        Initialize OpenAI client using environment variable
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        self.verbose = verbose
        self.max_input_len = max_input_len  # Not used directly but kept for compatibility

        # Default writer profile (system prompt)
        self.default_profile = """You are a seasoned writer who has won several accolades for your emotionally rich stories. 
            Your writing is renowned for painting vivid emotional landscapes, making readers not just observe 
            but truly feel the world of your characters."""

    def generate_story(self, 
                      premise: str, 
                      num_words: int = 500, 
                      profile: str = None,
                      examples: list = None,
                      temperature: float = 1.0):
        """
        Generate story using OpenAI API
        """
        start_time = time.time()
        try:
            # Construct messages
            messages = []
            system_prompt = profile or self.default_profile
            messages.append({"role": "system", "content": system_prompt})
            
            user_prompt = f"Write a {num_words}-word story about: {premise}\nRespond only with the story text."
            if examples:
                user_prompt += "\n\nExamples of good stories:\n"
                for ex in examples:
                    user_prompt += f"\nPremise: {ex['premise']}\nStory: {ex['story_excerpt']}\n"
            
            messages.append({"role": "user", "content": user_prompt})

            # API call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=int(num_words * 2),  # Rough estimate
            )

            story_text = response.choices[0].message.content.strip()
            generation_time = time.time() - start_time

            return {
                "story": story_text,
                "generation_time": generation_time,
                "usage": response.usage.dict() if response.usage else None
            }

        except Exception as e:
            print(f"OpenAI API error: {e}")
            traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(description='Generate multiple story versions per premise using OpenAI')
    parser.add_argument('--input_csv', type=str, default='./dataset/data/premises.csv',
                      help='Path to input CSV with premise_id and premise')
    parser.add_argument('--output_csv', type=str, default='openai_stories.csv',
                      help='Output CSV file for metadata')
    parser.add_argument('--output_dir', type=str, default='./dataset/data',
                      help='Directory for story files')
    parser.add_argument('--num_versions', type=int, default=1,
                      help='Number of story versions to generate per premise')
    parser.add_argument('--default_num_words', type=int, default=500,
                      help='Default word count for stories')
    parser.add_argument('--default_temperature', type=float, default=1.0,
                      help='Default temperature for generation')
    parser.add_argument('--max_retries', type=int, default=3,
                      help='Max regeneration attempts for word count validation')
    args = parser.parse_args()

    # Load premises
    try:
        df_premises = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error loading CSV: {traceback.format_exc()}")
        return

    # Configure models
    models_to_test = [
        {
            "model_id": "gpt-4.5-preview",
        },
    ]

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_csv)

    # Generation loop
    results = []
    for model_config in models_to_test:
        model_id = model_config["model_id"]
        print(f"\n{'-'*40}")
        print(f"Processing model: {model_id}")
        
        try:
            generator = OpenAIGenerator(
                model_id=model_config["model_id"],
                verbose=False
            )

            for _, premise_row in tqdm(df_premises.iterrows(), total=len(df_premises)):
                for story_id in range(1, args.num_versions + 1):
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    result_row = {
                        'premise_id': premise_row['premise_id'],
                        'story_id': story_id,
                        'text': None,
                        'model': model_id,
                        'timestamp': timestamp,
                        'generation_time': None,
                        'num_words': args.default_num_words,
                        'temperature': args.default_temperature,
                        'error': None,
                        'tokens_used': None
                    }

                    requested_words = args.default_num_words
                    lower_bound = requested_words * 0.9
                    upper_bound = requested_words * 1.1
                    best_story = None
                    best_metrics = None
                    error_msgs = []
                    success = False

                    for attempt in range(args.max_retries + 1):
                        try:
                            result = generator.generate_story(
                                premise=premise_row['premise'],
                                num_words=requested_words,
                                temperature=args.default_temperature
                            )

                            if result and result['story']:
                                story_text = result['story']
                                word_count = len(story_text.split())
                                
                                if word_count >= lower_bound and word_count <= upper_bound:
                                    best_story = story_text
                                    best_metrics = {
                                        'generation_time': result['generation_time'],
                                        'word_count': word_count,
                                        'tokens': result.get('usage', {}).get('total_tokens')
                                    }
                                    success = True
                                    break
                                else:
                                    best_story = story_text  # Keep last attempt
                                    error_msg = f"Attempt {attempt+1}: Word count {word_count}"
                                    error_msgs.append(error_msg)
                            else:
                                error_msg = f"Attempt {attempt+1}: Empty response"
                                error_msgs.append(error_msg)

                        except Exception as e:
                            error_msg = f"Attempt {attempt+1} error: {str(e)}"
                            error_msgs.append(error_msg)

                    # Update result row
                    if success:
                        result_row.update({
                            'text': best_story,
                            'generation_time': best_metrics['generation_time'],
                            'num_words': best_metrics['word_count'],
                            'tokens_used': best_metrics['tokens'],
                            'error': None
                        })
                    else:
                        result_row.update({
                            'error': "; ".join(error_msgs),
                            'text': best_story,
                            'tokens_used': None
                        })
                        if best_story:
                            result_row['num_words'] = len(best_story.split())

                    results.append(result_row)

                    # Save incremental results
                    pd.DataFrame(results).to_csv(save_path, index=False)

        except Exception as e:
            print(f"Error with model {model_id}: {traceback.format_exc()}")
            continue

    # Final save and reporting
    try:
        df_results = pd.DataFrame(results)
        print(f"\nSaved results to {save_path}")
        
        # Calculate statistics
        success_df = df_results[df_results['error'].isna()]
        error_df = df_results[df_results['error'].notna()]
        
        print(f"\nSummary:")
        print(f"Total premises: {len(df_premises)}")
        print(f"Versions per premise: {args.num_versions}")
        print(f"Successful generations: {len(success_df)}")
        print(f"Failed generations: {len(error_df)}")
        if len(error_df) > 0:
            print("\nError breakdown:")
            print(error_df['error'].value_counts().to_string())
            
    except Exception as e:
        print(f"Error finalizing results: {traceback.format_exc()}")

if __name__ == "__main__":
    main()