import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dataset.strategies.writer_profile import WriterProfileGenerator

# Transformers + BitsAndBytesConfig for 8‑bit + CPU offload
from transformers import BitsAndBytesConfig
# The Guidance “Transformers” wrapper
from guidance.models._transformers import Transformers as GuidanceTransformers

def main(args):
    # --- 1) Load premises CSV ---
    try:
        df_premises = pd.read_csv(args.input_csv)
    except UnicodeDecodeError:
        try:
            df_premises = pd.read_csv(args.input_csv, encoding='latin-1')
        except Exception:
            print("Error loading CSV:", traceback.format_exc())
            return
    except Exception:
        print("Error loading CSV:", traceback.format_exc())
        return

    # --- 2) Prepare the single, fixed model config ---
    MODEL_ID = "burtenshaw/GemmaCoder3-12B"
    quant_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # We pass a dummy load_in_8bit=False here so BaseStrategy won't try to 8‑bit load itself.
    model_config = {
        "backend_type": "transformers",
        "model_id": MODEL_ID,
        "load_in_8bit": False,
        "cache_dir": args.cache_dir,
        "device_map": args.device_map,
    }

    # --- 3) Prepare output directory & CSV path ---
    if not args.output_dir:
        args.output_dir = os.path.join('dataset/data', MODEL_ID.replace('/', '_'))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_csv)

    results = []

    try:
        # --- 4) Init your generator (but bypass its transformer‑loader) ---
        generator = WriterProfileGenerator(**model_config, verbose=False)

        # --- 5) Override guidance_model with an 8‑bit + CPU‑offload version ---
        generator.guidance_model = GuidanceTransformers(
            model=MODEL_ID,
            quantization_config=quant_cfg,
            device_map=args.device_map,
            cache_dir=args.cache_dir
        )

        # --- 6) Patch Gemma3Config so guidance doesn’t look for num_hidden_layers ---
        try:
            cfg = generator.guidance_model.model_obj.config
            if not hasattr(cfg, "num_hidden_layers"):
                cfg.num_hidden_layers = getattr(cfg, "n_layer", None) \
                                         or getattr(cfg, "num_layers", None)
        except Exception as e:
            print("⚠️ Warning patching num_hidden_layers:", e)

        print(f"\n{'='*40}")
        print(f"Using model {MODEL_ID} on transformers/8‑bit+offload")

        # --- 7) Main loop over premises + retries + bounds checking ---
        for _, premise_row in tqdm(df_premises.iterrows(), total=len(df_premises)):
            for story_id in range(1, args.num_versions + 1):
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                row = {
                    'premise_id': premise_row.get('premise_id'),
                    'story_id': story_id,
                    'text': None,
                    'backend': 'transformers',
                    'model': MODEL_ID,
                    'timestamp': ts,
                    'generation_time': None,
                    'num_words': args.default_num_words,
                    'temperature': args.default_temperature,
                    'error': None
                }

                target = args.default_num_words
                low, high = 0.8*target, 1.2*target

                best_text = None
                best_time = None
                best_count = 0
                errors = []
                success = False

                for attempt in range(args.max_retries + 1):
                    try:
                        out = generator.generate_story(
                            premise=premise_row['premise'],
                            num_words=target,
                            temperature=args.default_temperature,
                            profile=None,
                            examples=None
                        )
                        if not out:
                            msg = f"Attempt {attempt+1}: empty response"
                            print(msg); errors.append(msg)
                            continue

                        text = out['story']
                        count = len(text.split())
                        gen_time = out.get('generation_time')

                        if low <= count <= high:
                            best_text, best_time, best_count = text, gen_time, count
                            success = True
                            break
                        else:
                            msg = (f"Attempt {attempt+1}: word count {count} "
                                   f"not in [{low:.0f}, {high:.0f}]")
                            print(msg); errors.append(msg)
                            best_text, best_time, best_count = text, gen_time, count

                    except Exception as e:
                        msg = f"Attempt {attempt+1} error: {e}"
                        print(msg); errors.append(msg)

                # finalize this row
                if success:
                    row.update({
                        'text': best_text,
                        'generation_time': best_time,
                        'num_words': best_count
                    })
                else:
                    row['error'] = "; ".join(errors)
                    if best_text:
                        row.update({
                            'text': best_text,
                            'generation_time': best_time,
                            'num_words': best_count
                        })

                results.append(row)
                pd.DataFrame(results).to_csv(save_path, index=False)

    except Exception:
        print("Error in transformer run:", traceback.format_exc())

    # --- 8) Final summary (guarding against missing “error” column) ---
    if not results:
        print("No stories were generated.")
        return

    try:
        df_res = pd.DataFrame(results)
        print(f"\nSaved results to {save_path}")

        if 'error' in df_res.columns:
            succ = df_res[df_res['error'].isna()]
            fail = df_res[df_res['error'].notna()]
        else:
            succ, fail = df_res, df_res.iloc[0:0]  # treat all as successes if no errors

        print("\nSummary:")
        print(f" Total premises:       {len(df_premises)}")
        print(f" Versions per premise: {args.num_versions}")
        print(f" Successful runs:      {len(succ)}")
        print(f" Failed runs:          {len(fail)}")

        if not fail.empty:
            print("\nError breakdown:")
            print(fail['error'].value_counts().to_string())

    except Exception:
        print("Error finalizing results:", traceback.format_exc())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate story versions with GemmaCoder3‑12B (8‑bit + CPU offload)"
    )
    parser.add_argument('--input_csv', type=str, default='data/premises.csv',
                        help='Path to input CSV of premises')
    parser.add_argument('--output_csv', type=str, default='stories.csv',
                        help='Output CSV filename')
    parser.add_argument('--output_dir', type=str,
                        help='Where to save the output CSV')
    parser.add_argument('--num_versions', type=int, default=10,
                        help='How many stories per premise')
    parser.add_argument('--default_num_words', type=int, default=500,
                        help='Target word count')
    parser.add_argument('--default_temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--max_retries', type=int, default=5,
                        help='Retries on each story if bounds aren’t met')
    parser.add_argument('--cache_dir', type=str, default='~/.cache/hf',
                        help='HuggingFace cache directory')
    parser.add_argument('--device_map', type=str, default='auto',
                        help='Device map for model loading (e.g. "auto")')
    args = parser.parse_args()

    main(args)
