import os
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
import argparse
import traceback

from dataset.strategies.writer_profile import WriterProfileGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate multiple story versions per premise')
    parser.add_argument('--input_csv', type=str, default='./dataset/data/premises.csv',
                      help='Path to input CSV with premise_id and premise')
    parser.add_argument('--output_csv', type=str, default='llm_stories.csv',
                      help='Output CSV file for metadata')
    parser.add_argument('--output_dir', type=str, default='./dataset/data',
                      help='Directory for story files')
    parser.add_argument('--num_versions', type=int, default=3,
                      help='Number of story versions to generate per premise')
    parser.add_argument('--default_num_words', type=int, default=500,
                      help='Default word count for stories')
    parser.add_argument('--default_temperature', type=float, default=1.0,
                      help='Default temperature for generation')
    parser.add_argument('--max_retries', type=int, default=5,
                      help='Maximum number of regeneration attempts if word count is not within 10%% of the limit')
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
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "load_in_8bit": False,
            "cache_dir": "/data2/.shared_models"
        },
        # Add other models...
    ]

    # Create dir to save stories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_csv)

    # Generation loop
    results = []
    for model_config in models_to_test:
        model_id = model_config["model_id"]
        model_name = model_id.split('/')[-1]
        
        print(f"\n{'-'*40}")
        print(f"Processing model: {model_name}")
        
        try:
            generator = WriterProfileGenerator(
                model_id=model_config["model_id"],
                load_in_8bit=model_config["load_in_8bit"],
                cache_dir=model_config["cache_dir"],
                verbose=False
            )

            for _, premise_row in df_premises.iterrows():
                for story_id in range(1, args.num_versions + 1):
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    result_row = {
                        'premise_id': premise_row['premise_id'],
                        'story_id': story_id,
                        'text': None,
                        'model_id': model_id,
                        'timestamp': timestamp,
                        'generation_time': None,
                        'num_words': args.default_num_words,
                        'temperature': args.default_temperature,
                        'error': None
                    }

                    requested_words = args.default_num_words
                    lower_bound = requested_words * 0.9
                    upper_bound = requested_words * 1.1
                    best_story = None
                    best_generation_time = None
                    best_word_count = 0
                    error_msgs = []
                    success = False

                    for attempt in range(args.max_retries + 1):
                        try:
                            result = generator.generate_story(
                                premise=premise_row['premise'],
                                num_words=requested_words,
                                temperature=args.default_temperature,
                                profile=None,
                                examples=None
                            )

                            if result:
                                story_text = result['story']
                                word_count = len(story_text.split())
                                generation_time = result['generation_time']

                                if word_count >= lower_bound and word_count <= upper_bound:
                                    best_story = story_text
                                    best_generation_time = generation_time
                                    best_word_count = word_count
                                    success = True
                                    break  # Exit retry loop on success
                                else:
                                    # Keep the latest story even if invalid for potential error reporting
                                    best_story = story_text
                                    best_generation_time = generation_time
                                    best_word_count = word_count
                                    error_msg = f"Attempt {attempt + 1}: Word count {word_count} not within 10% of {requested_words}."
                                    error_msgs.append(error_msg)
                            else:
                                error_msg = f"Attempt {attempt + 1}: No story generated."
                                error_msgs.append(error_msg)

                        except Exception as e:
                            error_msg = f"Attempt {attempt + 1} error: {traceback.format_exc()}"
                            error_msgs.append(error_msg)

                    # Update result_row based on retry outcomes
                    if success:
                        result_row.update({
                            'text': best_story,
                            'generation_time': best_generation_time,
                            'num_words': best_word_count,
                            'error': None
                        })
                    else:
                        if best_story is not None:
                            # Save the story but note the word count error
                            result_row.update({
                                'text': best_story,
                                'generation_time': best_generation_time,
                                'num_words': best_word_count,
                                'error': f"Word count check failed after {args.max_retries} retries. " + "; ".join(error_msgs)
                            })
                        else:
                            # No story was generated in any attempt
                            result_row['error'] = "; ".join(error_msgs)

                    results.append(result_row)

                    # Save results incrementally
                    df_results = pd.DataFrame(results)
                    df_results.to_csv(save_path, index=False)

        except Exception as e:
            print(f"Error with model {model_name}: {traceback.format_exc()}")
            continue

    # Save results and report
    try:
        print(f"\nSaved results to {args.output_csv}")
        print(df_results)
        
        # Calculate statistics
        total_generated = df_results[df_results['error'].isna()].shape[0]
        total_errors = df_results[df_results['error'].notna()].shape[0]
        
        print(f"\nSummary:")
        print(f"Total premises processed: {len(df_premises)}")
        print(f"Versions per premise: {args.num_versions}")
        print(f"Models used: {len(models_to_test)}")
        print(f"Successfully generated stories: {total_generated}")
        print(f"Generation errors: {total_errors}")
        
    except Exception as e:
        print(f"Error saving results: {traceback.format_exc()}")

if __name__ == "__main__":

    # TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=7 python -m dataset.generate
    
    main()