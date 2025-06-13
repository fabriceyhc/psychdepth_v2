import pandas as pd
from math_util import compute_score

def clean(answer: str):
    answer = answer.replace('**', '').replace('__', '').rstrip('|.').strip()
    answer = answer.replace('I hope it is correct', '')
    answer = answer.removeprefix("The final answer is").lstrip(':').rstrip('.| ').strip()
    if '$\boxed' in answer:
        answer = answer.removeprefix('$\boxed{').removesuffix('}$')
    elif '$\\boxed' in answer:
        answer = answer.removeprefix('$\\boxed{').removesuffix('}$')
    try:
        answer = int(answer)
    except:
        print(answer)
        return None
    return answer

df = pd.read_csv("/data2/yihewang/psychdepth_v2/evaluate/results/gsm8k/gsm8k_answer_0shot.csv")
df.predicted_answer = df.predicted_answer.apply(clean)
df.dropna(subset=['predicted_answer'])
df.is_correct = df.apply(lambda row: 1 if (row['predicted_answer'] == row['answer']) else 0, axis=1)
