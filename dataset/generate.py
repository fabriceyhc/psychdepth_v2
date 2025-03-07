import os
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
import traceback
from tqdm import tqdm

from dataset.strategies.writer_profile import WriterProfileGenerator

def main(args):

    # Load premises
    try:
        df_premises = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error loading CSV: {traceback.format_exc()}")
        return

    # Configure models with different backends
    models_to_gen = [
        # Transformers backend example
        # {
        #     "backend_type": "transformers",
        #     "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        #     "load_in_8bit": False,
        #     "cache_dir": "/data2/.shared_models",
        #     "device_map": "auto"
        # },
        # OpenAI backend example
        {
            "backend_type": "openai",
            "model_id": "gpt-4.5-preview",
        },
        # Deepseek w/ OpenAI backend example
        # {
        #     "backend_type": "openai",
        #     "model_id": "deepseek-reasoner",
        #     "openai_base_url": "https://api.deepseek.com"
        # },
        # LlamaCpp backend example
        # {
        #     "backend_type": "llamacpp",
        #     "llamacpp_model_path": "/data2/.shared_models/llama.cpp_models/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",
        #     "llamacpp_n_ctx": 4096
        # }
    ]

    # Create dir to save stories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_csv)

    # Generation loop
    results = []
    for model_config in models_to_gen:
        backend_type = model_config["backend_type"]
        model_name = model_config.get("model_id", "") or Path(model_config.get("llamacpp_model_path", "")).stem
        
        print(f"\n{'-'*40}")
        print(f"Processing backend: {backend_type} | Model: {model_name}")
        
        try:
            # Initialize generator with current config
            generator = WriterProfileGenerator(
                **model_config,
                verbose=False
            )

            for _, premise_row in tqdm(df_premises.iterrows(), total=len(df_premises)):
                for story_id in range(1, args.num_versions + 1):
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    result_row = {
                        'premise_id': premise_row['premise_id'],
                        'story_id': story_id,
                        'text': None,
                        'backend': backend_type,
                        'model': model_name,
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

                                if lower_bound <= word_count <= upper_bound:
                                    best_story = story_text
                                    best_generation_time = generation_time
                                    best_word_count = word_count
                                    success = True
                                    break
                                else:
                                    best_story = story_text
                                    best_generation_time = generation_time
                                    best_word_count = word_count
                                    error_msg = f"Attempt {attempt+1}: Word count {word_count} not within range"
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
                            'generation_time': best_generation_time,
                            'num_words': best_word_count,
                            'error': None
                        })
                    else:
                        result_row['error'] = "; ".join(error_msgs)
                        if best_story:
                            result_row['text'] = best_story
                            result_row['generation_time'] = best_generation_time
                            result_row['num_words'] = best_word_count

                    results.append(result_row)

                    # Save incremental results
                    pd.DataFrame(results).to_csv(save_path, index=False)

        except Exception as e:
            print(f"Error with {backend_type} backend: {traceback.format_exc()}")
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
        print(f"Tested backends: {len(models_to_gen)}")
        print(f"Successful generations: {len(success_df)}")
        print(f"Failed generations: {len(error_df)}")
        if len(error_df) > 0:
            print("\nError breakdown:")
            print(error_df['error'].value_counts().to_string())
            
    except Exception as e:
        print(f"Error finalizing results: {traceback.format_exc()}")

if __name__ == "__main__":

    # python -m dataset.generate 

    import argparse

    parser = argparse.ArgumentParser(description='Generate multiple story versions per premise')
    parser.add_argument('--input_csv', type=str, default='./dataset/data/premises.csv',
                      help='Path to input CSV with premise_id and premise')
    parser.add_argument('--output_csv', type=str, default='GPT-4.5_stories.csv',
                      help='Output CSV file for metadata')
    parser.add_argument('--output_dir', type=str, default='./dataset/data',
                      help='Directory for story files')
    parser.add_argument('--num_versions', type=int, default=1,
                      help='Number of story versions to generate per premise')
    parser.add_argument('--default_num_words', type=int, default=500,
                      help='Default word count for stories')
    parser.add_argument('--default_temperature', type=float, default=1.0,
                      help='Default temperature for generation')
    parser.add_argument('--max_retries', type=int, default=5,
                      help='Maximum number of regeneration attempts if word count is not within 10%% of the limit')
    args = parser.parse_args()

    main(args)