import pandas as pd
from datasets import load_dataset
import json
import numpy as np

ds = load_dataset("allenai/UNcommonsense")

socialiqa_template = f"You are given a scenario (context) and an outcome. Your task is to explain why this outcome happened based on the context.\n\nContext: Cameron decided to have a barbecue and gathered her friends together.\nOutcome: Others feel bored and uninterested.\nExplanation of the outcome: Other than eating the food, there weren’t other activities planned.\n\nContext: Jan needed to give out jobs for an upcoming project at work.\nOutcome: Others will take a nap instead of working.\nExplanation of the outcome: The others don’t get paid more for doing the jobs Jan gave out.\n\nContext: Remy was an expert fisherman and was on the water with Kai. Remy baited Kai’s hook.\nOutcome: Remy will eat a sandwich.\nExplanation of the outcome: It’s been too long before they feel the weight of a fish, and Remy is hungry.\n\nContext: %s\nOutcome: %s\nExplanation of the outcome:"
rocstories_template = f"You are given a scenario (context) and an outcome. Your task is to explain why this outcome happened based on the context.\n\nContext: My friends all love to go to the club to dance. They think it’s a lot of fun and always invite. I finally decided to tag along last Saturday. I danced terribly and broke a friend’s toe.\nOutcome: My friends decided to keep inviting me out as I am so much fun.\nExplanation of the outcome: My friends thought the way I dance is really funny and they couldn’t stop laughing.\n\nContext: On the fourth of July, Lilly baked a lemon blueberry cake. She brought it to her boyfriend’s house and they had a bbq. After dinner they drove into the city to watch fireworks. When the show was over, they got donuts from a food truck.\nOutcome: Lilly had a terrible date.\nExplanation of the outcome: Lilly’s boyfriend kept complaining that the donuts were way better than the lemon blueberry cake she made, and her boyfriend just threw the cake away.\n\nContext: Jennifer was bored one Saturday. She decided to alleviate her boredom with a hike. She drove to a national park to go hiking. Jennifer hiked for hours.\nOutcome: Jennifer thought hiking was stupid.\nExplanation of the outcome: She realized the Saturday was a holiday, and the hiking trails in the national park were too crowded that it took her longer than usual to finish.\n\nContext: %s\nOutcome: %s\nExplanation of the outcome:"

sft_data = []
for i in ds['validation']:
    instruction_template = rocstories_template if i['source'] == 'rocstories' else socialiqa_template
    instruction = instruction_template % (i['context'], i['outcome'])
    for expl in i['enhanced_explanations']:
        sft_data.append({
            'conversations': [
                {
                    'from': 'human',
                    'value': instruction
                }
            ],
            'chosen': {
                'from': 'gpt',
                'value': expl
            },
            'rejected': {
                'from': 'gpt',
                'value': i['gpt4_explanations']
            }
        })
        
sft_data = np.random.permutation(sft_data)
dev_data = list(sft_data[:len(sft_data) // 2])
test_data = list(sft_data[len(sft_data) // 2:])
with open("dev_uncommonsense_dpo.json", "w") as f:
    json.dump(dev_data, f, indent=2)
with open("test_uncommonsense_dpo.json", "w") as f:
    json.dump(test_data, f, indent=2)