import re
import pandas as pd

for path in "Qwen2.5--baseline--GSM8k-eval.csv  Qwen2.5--GSM8k-answer--GSM8k-eval.csv  Qwen2.5--GSM8k-shuffled--GSM8k-eval.csv".split():
    df = pd.read_csv(path)

    def extract_integer(text):
        """Extracts an integer from a text string using layered heuristics."""
        text = str(text).split('<tool_call>')[0].strip()

        # Try direct conversion
        try:
            return int(text)
        except:
            pass

        # Try first token
        tokens = text.split()
        if tokens:
            try:
                return int(tokens[0])
            except:
                pass
        
        # Match "answer: 123"
        m = re.search(r'answer:\s*([0-9]+)', text.lower())
        if m:
            return int(m.group(1))

        # Match \boxed{123}
        m = re.match(r'\\\(\\boxed{([0-9]+)}', text)
        if m:
            return int(m.group(1))

        # General fallback: first number anywhere
        m = re.search(r'[0-9]+', text)
        if m:
            return int(m.group(0))

        # Fallback if no integer is found
        return -10**9

    def normalize_prediction(row):
        """Post-processes a predicted answer, checking for padding or numeric approximations."""
        pred = str(row['predicted_answer']).strip()
        ans = str(row['answer']).strip()

        if len(pred) > 10:
            # Case: predicted value starts with correct digits and is zero-padded
            if pred.startswith(ans) and all(c == '0' for c in pred[len(ans):10]):
                return int(ans)

            # Case: float approximation that is close to an int
            try:
                val = float(pred)
                if val.is_integer():
                    return int(val)
            except:
                pass

            # Final fallback for long predicted numbers
            try:
                return int(pred)
            except:
                return -10**9
        else:
            # Regular integer conversion
            try:
                return int(pred)
            except:
                return -10**9

    # --- Cleaning Pipeline ---

    # Fill missing predicted values with a dummy string
    df = df.fillna('-10000000')

    # Strip outer single quotes and whitespace
    df['predicted_answer'] = df['predicted_answer'].str.strip("'").str.strip()

    # Apply initial integer extraction
    df['predicted_answer'] = df['predicted_answer'].map(extract_integer)

    # Normalize predictions using answer context
    df['predicted_answer'] = df.apply(normalize_prediction, axis=1)

    # Compute accuracy
    df['is_correct'] = (df['predicted_answer'].astype(str) == df['answer'].astype(str)).astype(int)
    accuracy = df['is_correct'].mean()
    print(accuracy)
    df.to_csv(path, index=False)