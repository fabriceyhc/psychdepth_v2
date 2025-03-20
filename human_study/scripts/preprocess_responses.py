import pandas as pd
import glob
import re
from functools import reduce

###############################################################
# Helper Functions
###############################################################

def csv_to_df(csv_files, mappings):
    """
    Reads a list of CSV files and renames columns based on mappings.
    Each CSV file's Participant ID column is renamed to include an index.
    Also, for each mapping (e.g. "**authentic**" to "authenticity"),
    all matching columns get a prefix (like "0." or "1.").
    """
    dfs = []
    for i, f in enumerate(csv_files):
        df = pd.read_csv(f)
        # Find the participant ID column robustly (ignoring line ending differences)
        p_id_candidates = [col for col in df.columns if col.lower().startswith("participant id")]
        if not p_id_candidates:
            raise KeyError(f"Participant ID column not found in file: {f}")
        p_id = p_id_candidates[0]
        # Rename the participant ID column to include file index
        df = df.rename(columns={p_id: f"{i}.participant_id"})
        # Rename columns based on the provided mappings.
        for search_val, rename_val in mappings:
            target_columns = [col for col in df.columns if search_val in col.lower()]
            renamed_columns = [f"{i}.{col}" for col in target_columns]
            df = df.rename(columns=dict(zip(target_columns, renamed_columns)))
        # Ensure the participant id column exists
        pid_col = f"{i}.participant_id"
        if pid_col not in df.columns:
            raise KeyError(f"Expected participant id column {pid_col} not found in file: {f}")
        df = df.sort_values(by=pid_col)
        df["order"] = i
        dfs.append(df)
    # Concatenate files vertically (stack rows)
    return pd.concat(dfs, axis=0)

def extract_story_index(col_name):
    """
    Extracts a story index from a column name.
    For example, if col_name ends with "?.<num>" (like "?.1"), returns int(num)+1.
    Otherwise, returns 1 (the base column is for story 1).
    """
    m = re.search(r'\?\.(\d+)$', col_name)
    if m:
        return int(m.group(1)) + 1
    else:
        return 1

def reform_melt(df, mappings):
    """
    For each rating category in mappings, melt the corresponding columns,
    extract a story index (study_id) from the column names,
    and then merge all the melted dataframes on participant_id and study_id.
    """
    melted_dfs = []
    for search_val, rename_val in mappings:
        # Get all columns that match the rating keyword.
        target_columns = [col for col in df.columns if search_val in col.lower()]
        if not target_columns:
            print(f"Warning: No columns found for search term {search_val}")
            continue

        # We'll use the first participant id column (assumed to be "0.participant_id")
        participant_id_col = [col for col in df.columns if col.endswith("participant_id")][0]
        # Subset the dataframe
        target_df = df[target_columns + [participant_id_col]]
        
        # Melt the dataframe so that each rating becomes a row
        df_melted = pd.melt(
            target_df, 
            id_vars=[participant_id_col],
            value_vars=target_columns,
            var_name=f'{rename_val}_source',
            value_name=f'{rename_val}_score'
        )
        # Rename the participant id column to a generic name
        df_melted = df_melted.rename(columns={participant_id_col: "participant_id"})
        # Extract the study_id from the melted source column
        df_melted['study_id'] = df_melted[f'{rename_val}_source'].apply(extract_story_index)
        # Drop the source column (optional)
        df_melted = df_melted.drop(columns=[f'{rename_val}_source'])
        # Set index to participant_id and study_id for merging
        df_melted = df_melted.set_index(['participant_id', 'study_id'])
        melted_dfs.append(df_melted)
    
    # Merge all melted dataframes on participant_id and study_id
    if not melted_dfs:
        raise ValueError("No melted dataframes to merge!")
    merged = reduce(lambda left, right: left.join(right, how='outer'), melted_dfs)
    merged = merged.reset_index()
    return merged

def filter_out_values(df, column, values_to_filter):
    """
    Filters out rows where a column has values that are in the blacklist.
    """
    mask = ~df[column].isin(values_to_filter)
    return df[mask]

###############################################################
# Load and Process Data
###############################################################

if __name__ == "__main__":

    # Get list of CSV files (using glob)
    csv_files = glob.glob("./human_study/data/Deepseek Story Annotation Study (Responses) - Form Responses *.csv")
    
    # Define the mappings from keywords to our simplified rating names.
    mappings = [
        ("**authentic**", "authenticity"),
        ("**empathy**", "empathy"),
        ("**engaging**", "engagement"),
        ("**provoke emotion**", "emotion_provoking"),
        ("**narratively complex**", "narrative_complexity"),
    ]
    
    # Convert CSV files to a single dataframe.
    df = csv_to_df(csv_files, mappings)
    
    # Melt the dataframe into a long format where each row corresponds to one story rating per participant.
    df_long = reform_melt(df, mappings)
    
    # Rearrange columns as desired.
    final_cols = ["participant_id", "study_id", 
                  "authenticity_score", "empathy_score", "engagement_score", 
                  "emotion_provoking_score", "narrative_complexity_score"]
    final_df = df_long[final_cols]
    
    
    # Display the result
    print(final_df.sort_values(["participant_id", "study_id"]).reset_index(drop=True))
    print(f"Total rows: {len(final_df)}")

    save_path = "./human_study/data/processed_responses.csv"
    final_df.to_csv(save_path, index=False)
    print(f"Processed responses saved to: {save_path}")

    df_stories = pd.read_csv("./human_study/data/deepseek_stories.csv")
    df_stories = df_stories.set_index("study_id")

    finaler_df = final_df.join(df_stories[["premise", "text", "author_short"]], on='study_id', how='left', lsuffix='', rsuffix='_story')

    print(finaler_df)

    save_path = "./human_study/data/processed_responses_w_metadata.csv"
    finaler_df.to_csv(save_path, index=False)
    print(f"Processed responses saved to: {save_path}")