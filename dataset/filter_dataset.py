import argparse
import os
import sys
from datasets import load_dataset, Dataset, DatasetDict, Features
from typing import Union, Dict, Any 
import math

def filter_dataset_by_source(
    dataset_obj: Union[Dataset, DatasetDict],
    source_value: str
) -> Union[Dataset, DatasetDict]:
    """
    Filters a Hugging Face Dataset or DatasetDict to include only rows
    where the 'source' column matches the specified source_value.
    Uses batched filtering for efficiency.

    Args:
        dataset_obj: The Hugging Face Dataset or DatasetDict to filter.
                     It must contain a column named 'source'.
        source_value: The string value to filter for in the 'source' column.

    Returns:
        A new Dataset or DatasetDict containing only the filtered rows.
        Returns the same type as the input (Dataset or DatasetDict).

    Raises:
        ValueError: If the input object is not a Dataset or DatasetDict.
        KeyError: If the 'source' column is missing in the dataset features.
    """
    if not isinstance(dataset_obj, (Dataset, DatasetDict)):
        raise ValueError("Input must be a Hugging Face Dataset or DatasetDict.")

    # --- Check for 'source' column before filtering ---
    dataset_features: Features
    if isinstance(dataset_obj, Dataset):
        dataset_features = dataset_obj.features
        if 'source' not in dataset_features:
             raise KeyError("The input Dataset does not have a 'source' column in its features.")
    elif isinstance(dataset_obj, DatasetDict):
         if not dataset_obj: raise ValueError("Input DatasetDict is empty.")
         first_split_name = next(iter(dataset_obj.keys()))
         dataset_features = dataset_obj[first_split_name].features
         if 'source' not in dataset_features:
            raise KeyError(f"The '{first_split_name}' split (and presumably others) of the input DatasetDict does not have a 'source' column in its features.")

    # --- Define the filtering function FOR BATCHED MODE ---
    def is_source_match_batched(batch):
        sources_in_batch = batch.get('source')
        if sources_in_batch is None:
             if not batch: return []
             try:
                 first_col_key = next(iter(batch.keys()))
                 batch_size = len(batch[first_col_key])
             except (StopIteration, KeyError, TypeError):
                 batch_size = 0
             return [False] * batch_size
        return [source == source_value for source in sources_in_batch]

    # --- Apply the filter ---
    print(f"\nFiltering dataset(s) to keep rows where 'source' == '{source_value}'...")
    filtered_dataset_obj = dataset_obj.filter(
        is_source_match_batched,
        batched=True
    )
    # Note: If input was DatasetDict, filter is applied to all splits.
    # The user's loading logic `['train']` ensures we deal with a Dataset here.
    print("Filtering complete.")
    return filtered_dataset_obj

def main():
    parser = argparse.ArgumentParser(
        description="Filter a Hugging Face dataset based on the 'source' column and optionally split it."
    )
    # Filtering Args
    parser.add_argument(
        "--source_value",
        type=str,
        required=True, # Made required again, as default might not be intended
        help="The value in the 'source' column to filter for."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="open-r1/OpenR1-Math-220k",
        help="Name of the dataset on HF Hub or path to a local dataset directory. "
             "Defaults to 'open-r1/OpenR1-Math-220k'."
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="default",
        help="Configuration name for the dataset (if needed). Defaults to 'default'."
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Which split of the original dataset to load and filter. Defaults to 'train'."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data2/.shared_datasets/", # Default to standard HF cache
        help="Directory for caching Hugging Face datasets."
    )
    # Splitting Args
    parser.add_argument(
        "--perform_split",
        action='store_true', # Makes it a flag: presence means True, absence means False
        help="If set, perform train/validation/test splitting on the filtered data."
    )
    parser.add_argument(
        "--test_size",
        type=float, # Accepts float or int >= 1.0 (interpreted as count)
        default=0.1,
        help="Proportion (float < 1.0) or absolute number (int >= 1) for the test set "
             "relative to the filtered data. Defaults to 0.1 (10%)."
    )
    parser.add_argument(
        "--val_size",
        type=float, # Accepts float or int >= 1.0 (interpreted as count)
        default=0.1,
        help="Proportion (float < 1.0) or absolute number (int >= 1) for the validation set "
             "relative to the data *remaining after the test split*. Defaults to 0.1 (10% of remainder)."
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for train/val/test splitting. Defaults to 42."
    )
    # Output Arg
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional. Path to a directory where the final dataset(s) will be saved. "
             "If splitting, saves a DatasetDict; otherwise saves the filtered data under the 'train' key."
    )

    args = parser.parse_args()

    # --- 1. Load Original Data ---
    print(f"Attempting to load dataset: {args.dataset_name} (config: {args.dataset_config}, split: {args.dataset_split})")
    try:
        # Load the specified split of the dataset
        original_dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            cache_dir=args.cache_dir,
            split=args.dataset_split # Directly load the desired split
        )
        print("Dataset split loaded successfully.")
        print(f"\nOriginal '{args.dataset_split}' split Structure:")
        print(original_dataset)
        original_count = len(original_dataset)

    except Exception as e:
        print(f"\nError loading dataset split '{args.dataset_split}' from '{args.dataset_name}': {e}", file=sys.stderr)
        print("Please ensure the dataset name/config/split/path is correct and you have access.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Filter Data ---
    try:
        filtered_dataset = filter_dataset_by_source(original_dataset, args.source_value)
        filtered_count = len(filtered_dataset)
        print(f"\nFiltered dataset has {filtered_count} rows (originally {original_count} in '{args.dataset_split}' split).")

        if filtered_count == 0:
             print("\nFiltered dataset is empty. No splitting or saving will occur.", file=sys.stderr)
             sys.exit(0) # Exit normally, nothing more to do

    except (ValueError, KeyError) as e:
        print(f"\nError during filtering process: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during filtering: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Split Data (if requested) ---
    final_output_dict: Dict[str, Dataset] = {} # Type hint for clarity
    if args.perform_split:
        print(f"\nPerforming train/validation/test split with seed {args.split_seed}...")
        print(f"  Target test size: {args.test_size}")
        print(f"  Target validation size (of remainder): {args.val_size}")

        try:
            # First split: Separate test set
            # Ensure test_size is valid for the dataset size
            actual_test_size = args.test_size
            if isinstance(actual_test_size, float) and actual_test_size >= 1.0:
                 actual_test_size = int(actual_test_size) # Convert counts to int
            if isinstance(actual_test_size, int) and actual_test_size >= filtered_count:
                 raise ValueError(f"test_size ({actual_test_size}) is >= filtered dataset size ({filtered_count}). Cannot split.")
            if isinstance(actual_test_size, float) and (actual_test_size <= 0.0 or actual_test_size >= 1.0):
                  raise ValueError(f"test_size proportion ({actual_test_size}) must be between 0.0 and 1.0 (exclusive).")


            trainval_split = filtered_dataset.train_test_split(
                test_size=actual_test_size,
                seed=args.split_seed,
                shuffle=True # Good practice to shuffle before splitting
            )
            test_set = trainval_split['test']
            remaining_for_trainval = trainval_split['train']
            print(f"  Split off test set: {len(test_set)} rows")

            if len(remaining_for_trainval) == 0:
                 print("  No data remaining after test split. Only 'test' split created.", file=sys.stderr)
                 final_output_dict = {'test': test_set}
            else:
                 # Second split: Separate validation set from the remainder
                 actual_val_size = args.val_size
                 if isinstance(actual_val_size, float) and actual_val_size >= 1.0:
                     actual_val_size = int(actual_val_size)
                 if isinstance(actual_val_size, int) and actual_val_size >= len(remaining_for_trainval):
                     raise ValueError(f"val_size ({actual_val_size}) is >= remaining data size ({len(remaining_for_trainval)}). Cannot split.")
                 if isinstance(actual_val_size, float) and (actual_val_size <= 0.0 or actual_val_size >= 1.0):
                      raise ValueError(f"val_size proportion ({actual_val_size}) must be between 0.0 and 1.0 (exclusive).")


                 train_final_split = remaining_for_trainval.train_test_split(
                     test_size=actual_val_size, # val_size applied to the remainder
                     seed=args.split_seed, # Use same seed for reproducibility chain
                     shuffle=True
                 )
                 train_set = train_final_split['train']
                 val_set = train_final_split['test'] # The 'test' part of this split is our validation set
                 print(f"  Split remainder into train ({len(train_set)} rows) and validation ({len(val_set)} rows)")

                 final_output_dict = {
                     'train': train_set,
                     'validation': val_set,
                     'test': test_set
                 }

        except ValueError as e:
             print(f"\nError during splitting: {e}", file=sys.stderr)
             print("Splitting aborted. Check sizes relative to filtered data.", file=sys.stderr)
             # Decide if you want to exit or proceed without splitting
             # For now, let's clear the dict and proceed to save nothing or just filtered
             final_output_dict = {} # Indicate splitting failed
             print("Proceeding without split data.")
             # Alternatively, re-assign filtered data if saving is desired even on split failure:
             # final_output_dict = {'train': filtered_dataset}
             # print("Proceeding with unsplit filtered data.")
             # Or simply exit: sys.exit(1)
        except Exception as e:
             print(f"\nAn unexpected error occurred during splitting: {e}", file=sys.stderr)
             sys.exit(1)


    else:
        # No splitting requested, prepare output dict with the single filtered dataset
        print("\nSplitting not requested. Using filtered data as 'train' split.")
        final_output_dict = {'train': filtered_dataset}

    # --- 4. Report Final Sizes and Save ---
    if final_output_dict:
        final_output_datasetdict = DatasetDict(final_output_dict)
        print("\nFinal Dataset Structure:")
        print(final_output_datasetdict)

        print("\nFinal row counts per split:")
        for split_name, dataset in final_output_datasetdict.items():
            print(f"  {split_name}: {len(dataset)} rows")

        # --- Save the final dataset(s) if an output directory is specified ---
        if args.output_dir:
            # Define the desired max shard size (use string for readability)
            max_shard_size_str = "50MB"
            print(f"\nSaving final dataset(s) to: {args.output_dir} with max shard size {max_shard_size_str}")
            try:
                os.makedirs(args.output_dir, exist_ok=True)
                # *** MODIFICATION HERE ***
                final_output_datasetdict.save_to_disk(
                    args.output_dir,
                    max_shard_size=max_shard_size_str # Control shard size
                )
                # *************************
                print("Dataset(s) saved successfully.")
                print(f"  Check '{args.output_dir}' for dataset files (multiple .arrow files per split expected).")
            except Exception as e:
                print(f"Error saving dataset(s) to '{args.output_dir}': {e}", file=sys.stderr)
    else:
         print("\nNo final dataset generated (possibly due to empty filtered data or splitting error).")


if __name__ == "__main__":

    # python -m dataset.filter_dataset --source_value amc_aime --perform_split --test_size 100 --val_size 50 --output_dir ./data/open-r1/OpenR1-Math-220k/amc_aime
    # python -m dataset.filter_dataset --source_value olympiads --perform_split --test_size 100 --val_size 50 --output_dir ./data/open-r1/OpenR1-Math-220k/olympiads

    # from datasets import load_from_disk
    # loaded_data = load_from_disk("./data/open-r1/OpenR1-Math-220k/amc_aime")
    # loaded_data = load_from_disk("./data/open-r1/OpenR1-Math-220k/olympiads")

    main()