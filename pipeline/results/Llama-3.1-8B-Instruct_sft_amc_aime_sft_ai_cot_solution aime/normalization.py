import pandas as pd
import re

def extract_options_mapping(problem_text):
    """Extract mapping between letters (A,B,C,D,E) and their corresponding values"""
    # Look for options in format "\textbf{(A)}\ 1" or "(A) 1"
    pattern = r'(?:\\textbf\{)?\(?([A-E])\)?(?:\}\s*\\)?[^\w]*(\d+|\\frac\{[^}]+\}\{[^}]+\})'
    matches = re.findall(pattern, problem_text)
    
    # Create a dictionary mapping values to letters
    value_to_letter = {}
    for letter, value in matches:
        value_to_letter[value] = letter
        
        # Handle numeric values as well
        if value.isdigit():
            value_to_letter[value] = letter
    
    return value_to_letter

def extract_letter_from_boxed(answer_text):
    """Extract letter from boxed format like \\boxed{C}"""
    boxed_match = re.search(r'\\boxed\{([A-E])\}', answer_text)
    if boxed_match:
        return boxed_match.group(1)
    return None

def extract_number_from_boxed(answer_text):
    """Extract number from boxed format like \\boxed{3}"""
    boxed_match = re.search(r'\\boxed\{(\d+)\}', answer_text)
    if boxed_match:
        return boxed_match.group(1)
    return None

def normalize_answer(row):
    """Normalize answers for comparison"""
    predicted = str(row['predicted_answer']).strip()
    answer = str(row['answer']).strip()
    
    # First check for direct letter answer
    if len(answer) == 1 and answer in 'ABCDE':
        normalized_answer = answer
    else:
        # This is a numeric or symbolic answer
        normalized_answer = answer
    
    # Check for boxed letter in predicted answer
    letter_match = extract_letter_from_boxed(predicted)
    if letter_match:
        normalized_prediction = letter_match
    else:
        # Check for boxed number in predicted answer
        number_match = extract_number_from_boxed(predicted)
        if number_match:
            normalized_prediction = number_match
        else:
            # If not in boxed format, use the predicted answer as is
            normalized_prediction = predicted
    
    return {
        'normalized_answer': normalized_answer,
        'normalized_prediction': normalized_prediction
    }

def letter_number_mapping(problems):
    """Build a dictionary mapping problem IDs to their option mappings"""
    mappings = {}
    
    for idx, problem_text in enumerate(problems):
        try:
            # Extract the letter-to-value mapping
            mapping = extract_options_mapping(problem_text)
            if mapping:
                mappings[idx] = mapping
        except Exception as e:
            print(f"Error processing problem {idx}: {e}")
    
    return mappings

def process_csv(file_path):
    """Process CSV file and add normalized answers"""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Apply normalization to each row
    normalized_data = df.apply(normalize_answer, axis=1)
    
    # Create new columns from the normalized data
    df['normalized_answer'] = normalized_data.apply(lambda x: x['normalized_answer'])
    df['normalized_prediction'] = normalized_data.apply(lambda x: x['normalized_prediction'])
    
    # Find answers with letter format (A,B,C,D,E)
    letter_pattern = r'^[A-E]$'
    df['answer_is_letter'] = df['normalized_answer'].str.match(letter_pattern, na=False)
    
    # Add letter mapping for answers that are numbers
    # For problems with option mappings like (C) 3
    option_mappings = letter_number_mapping(df['problem'])
    
    # Convert boxed answers to corresponding letter when appropriate
    for idx, row in df.iterrows():
        # If answer is a letter and prediction is a number that corresponds to a letter
        if row['answer_is_letter'] and row['normalized_prediction'].isdigit():
            if idx in option_mappings and row['normalized_prediction'] in option_mappings[idx]:
                df.at[idx, 'normalized_prediction'] = option_mappings[idx][row['normalized_prediction']]
        
        # If answer is a number and prediction is a letter
        elif not row['answer_is_letter'] and row['normalized_prediction'] in 'ABCDE':
            # Check if there's a mapping from letter to number
            letter_matches = []
            if idx in option_mappings:
                letter_matches = [k for k, v in option_mappings[idx].items() if v == row['normalized_prediction']]
            
            if letter_matches:
                df.at[idx, 'normalized_prediction'] = letter_matches[0]
    
    # Compare the normalized answers
    df['is_correct_normalized'] = df['normalized_answer'] == df['normalized_prediction']
    
    # Drop the temporary column
    df = df.drop('answer_is_letter', axis=1)
    
    return df

# Usage example
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "Llama-3.1-8B-Instruct_sft_0shot_old.csv"
    result_df = process_csv(file_path)
    result_df.to_csv("normalized_Llama-3.1-8B-Instruct_sft_0shot_old.csv", index=False)
    print(f"Processed {len(result_df)} rows. Normalized answers saved to normalized_math_answers.csv")