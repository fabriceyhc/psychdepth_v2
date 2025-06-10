import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import traceback
from tqdm import tqdm

from dataset.strategies.writer_profile import WriterProfileGenerator

def main(args):
    # Load premises
    try:
        # Try reading with UTF-8 first
        df_premises = pd.read_csv(args.input_csv)
    except UnicodeDecodeError:
        try:
            # Common fallback encodings
            df_premises = pd.read_csv(args.input_csv, encoding='latin-1')
        except Exception as e:
            print(f"Error loading CSV: {traceback.format_exc()}")
            return
    except Exception as e:
        print(f"Error loading CSV: {traceback.format_exc()}")
        return

    # Create model config from arguments
    model_config = {
        "backend_type": args.backend_type,
        "model_id": args.model_id,
        "load_in_8bit": args.load_in_8bit,
        "cache_dir": args.cache_dir,
        "device_map": args.device_map,
        "openai_base_url": args.openai_base_url,
        "llamacpp_model_path": args.llamacpp_model_path,
        "llamacpp_n_ctx": args.llamacpp_n_ctx
    }

    # Clean up empty values
    model_config = {k: v for k, v in model_config.items() if v is not None}

    backend_type = model_config["backend_type"]
    model_name = model_config.get("model_id", "") or \
                    Path(model_config.get("llamacpp_model_path", "")).stem

    # Create output directory
    if not args.output_dir:
        args.output_dir = os.path.join('./dataset/data/', model_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_csv)

    results = []
    
    try:
        # Initialize generator
        generator = WriterProfileGenerator(**model_config, verbose=False)

        print(f"\n{'-'*40}")
        print(f"Processing backend: {backend_type} | Model: {model_name}")

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
                lower_bound = requested_words * 0.8
                upper_bound = requested_words * 1.5
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
                            elif best_story != None:
                                if word_count < len(best_story.split()):
                                    best_story = story_text
                                    best_generation_time = generation_time
                                    best_word_count = word_count
                                    error_msg = f"Attempt {attempt+1}: Word count {word_count} not within range"
                                    print(error_msg)
                                    error_msgs.append(error_msg)
                            else:
                                best_story = story_text
                                best_generation_time = generation_time
                                best_word_count = word_count
                                error_msg = f"Attempt {attempt+1}: Word count {word_count} not within range"
                                print(error_msg)
                                error_msgs.append(error_msg)
                        else:
                            error_msg = f"Attempt {attempt+1}: Empty response"
                            print(error_msg)
                            error_msgs.append(error_msg)

                    except Exception as e:
                        error_msg = f"Attempt {attempt+1} error: {str(e)}"
                        print(error_msg)
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
                pd.DataFrame(results).to_csv(save_path, index=False)

    except Exception as e:
        print(f"Error with {backend_type} backend: {traceback.format_exc()}")

    # Final save and reporting
    try:
        df_results = pd.DataFrame(results)
        print(f"\nSaved results to {save_path}")
        
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
    import argparse

    parser = argparse.ArgumentParser(description='Generate story versions with a single model configuration')
    
    # Core arguments
    parser.add_argument('--input_csv', type=str, default='./data/premises.csv',
                      help='Path to input CSV with premises')
    parser.add_argument('--output_csv', type=str, default='stories.csv',
                      help='Output CSV filename')
    parser.add_argument('--output_dir', type=str,
                      help='Output directory')
    
    # Generation parameters
    parser.add_argument('--num_versions', type=int, default=1,
                      help='Number of versions per premise')
    parser.add_argument('--default_num_words', type=int, default=500,
                      help='Target word count for stories')
    parser.add_argument('--default_temperature', type=float, default=1.0,
                      help='Generation temperature')
    parser.add_argument('--max_retries', type=int, default=5,
                      help='Max retry attempts per story')
    
    # Model configuration arguments
    parser.add_argument('--backend_type', type=str, required=True,
                      choices=['openai', 'transformers', 'llamacpp'],
                      help='Backend type for model inference')
    parser.add_argument('--model_id', type=str,
                      help='Model identifier (for OpenAI/Transformers backends)')
    parser.add_argument('--load_in_8bit', action='store_true',
                      help='Load model in 8-bit mode (Transformers backend)')
    parser.add_argument('--cache_dir', type=str,
                      help='Model cache directory (Transformers backend)')
    parser.add_argument('--device_map', type=str, default="auto",
                      help='Device map for model loading (Transformers backend)')
    parser.add_argument('--openai_base_url', type=str,
                      help='Custom base URL for OpenAI API')
    parser.add_argument('--llamacpp_model_path', type=str,
                      help='Path to GGUF model file (LlamaCpp backend)')
    parser.add_argument('--llamacpp_n_ctx', type=int, default=4096,
                      help='Context size for LlamaCpp backend')

    args = parser.parse_args()

    # Validation
    if args.backend_type == 'llamacpp' and not args.llamacpp_model_path:
        raise ValueError("--llamacpp_model_path is required for LlamaCpp backend")
    
    if args.backend_type == 'openai' and not args.model_id:
        args.model_id = 'gpt-4o'  # Set default OpenAI model
    
    main(args)

# # OpenAI example
# python -m dataset.generate_stories \
#   --backend_type openai \
#   --model_id gpt-4-turbo \
#   --openai_base_url https://api.example.com/v1 \
#   --num_versions 3 \
#   --output_csv openai_stories.csv

# # Deepseek example
# python -m dataset.generate_stories \
#   --backend_type openai \
#   --model_id deepseek-reasoner \
#   --openai_base_url https://api.deepseek.com

# # Transformers example
# CUDA_VISIBLE_DEVICES=0 python -m dataset.generate_stories \
#   --backend_type transformers \
#   --model_id meta-llama/Llama-3.2-1B-Instruct

# # LlamaCpp example
# python -m dataset.generate_stories \
#   --backend_type llamacpp \
#   --llamacpp_model_path /data2/.shared_models/llama.cpp_models/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf \
#   --llamacpp_n_ctx 8192