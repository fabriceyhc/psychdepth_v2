# psychdepth_v2

## Clone the repository
```
git clone https://github.com/fabriceyhc/psychdepth_v2.git
cd psychdepth_v2
```

## Environment Setup
```
pip install -r requirements.txt
pip install --no-build-isolation flash-attn
```

Next, set up LLaMA-Factory dependencies
```
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## File Links
```
Since the json files created for the codeforces dataset were huge (>500 MG), I uploaded the codeforces_ai_cots.json and codeforces_human_cots.json here:
https://drive.google.com/drive/folders/1L7ns3li3sPiw8sIbHe9r9MZKq2sTp1XR?usp=sharing
```


## Generate stories
```
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=X python -m dataset.generate 
```
where X is an open GPU node ID (int)

## Story Evaluation

To analyze stories for psychological depth, you can run one of the following commands depending on whether you want to run a local model or openai. Local models rely on [guidance](https://github.com/guidance-ai/guidance), an excellent framework for controlling LLMs. Guidance works best when it has access to the token probabilities of a model so we only used it for Llama-3. 

```
evaluator = PsychDepthEvaluator(model_id="meta-llama/Llama-3.2-3B-Instruct")

story = "Once upon a time, there was a brave knight who..."
results = evaluator.evaluate(story=story, temperature=1.0)
print(f"results: {results}")

results_with_personas = evaluator.evaluate(story=story, personas=evaluator.personas, temperature=1.0)
print(f"results_with_personas: {results_with_personas}")
```

```
## results:
{
    "persona_0": {
        "authenticity_score": 2.0,
        "emotion_provoking_score": 3.0,
        "empathy_score": 5.0,
        "engagement_score": 3.0,
        "narrative_complexity_score": 4.0,
        "human_likeness_score": 3.0,
        "persona_id": 0,
        "persona": None,
        "time_taken": 1.17307448387146,
    }
}

## results_with_personas: 
{
    "persona_0": {
        "authenticity_score": 4.0,
        "emotion_provoking_score": 4.0,
        "empathy_score": 3.0,
        "engagement_score": 4.0,
        "narrative_complexity_score": 5.0,
        "human_likeness_score": 2.0,
        "persona_id": 0,
        "persona": "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you specialize in evaluating the genuineness and believability of characters, dialogue, and scenarios in stories.",
        "time_taken": 0.777904748916626,
    },
    "persona_1": {
        "authenticity_score": 2.0,
        "emotion_provoking_score": 4.0,
        "empathy_score": 4.0,
        "engagement_score": 4.0,
        "narrative_complexity_score": 3.0,
        "human_likeness_score": 5.0,
        "persona_id": 1,
        "persona": "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you focus on identifying and assessing moments in the narrative that effectively evoke empathetic connections with the characters.",
        "time_taken": 0.7759261131286621,
    },
    "persona_2": {
        "authenticity_score": 2.0,
        "emotion_provoking_score": 4.0,
        "empathy_score": 4.0,
        "engagement_score": 4.0,
        "narrative_complexity_score": 3.0,
        "human_likeness_score": 2.0,
        "persona_id": 2,
        "persona": "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you evaluate how well a story captures and maintains the reader's interest through pacing, suspense, and narrative flow.",
        "time_taken": 0.7819526195526123,
    },
    "persona_3": {
        "authenticity_score": 2.0,
        "emotion_provoking_score": 4.0,
        "empathy_score": 3.0,
        "engagement_score": 4.0,
        "narrative_complexity_score": 3.0,
        "human_likeness_score": 2.0,
        "persona_id": 3,
        "persona": "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you examine the text for its ability to provoke a wide range of intense emotional responses in the reader.",
        "time_taken": 0.7773287296295166,
    },
    "persona_4": {
        "authenticity_score": 3.0,
        "emotion_provoking_score": 4.0,
        "empathy_score": 4.0,
        "engagement_score": 4.0,
        "narrative_complexity_score": 3.0,
        "human_likeness_score": 2.0,
        "persona_id": 4,
        "persona": "You are a helpful AI who specializes in evaluating the psychological depth present in stories. In particular, you analyze the structural and thematic intricacy of the plot, character development, and the use of literary devices.",
        "time_taken": 0.7846338748931885,
    },
    "average": {
        "authenticity_score": 2.6,
        "emotion_provoking_score": 4.0,
        "empathy_score": 3.6,
        "engagement_score": 4.0,
        "narrative_complexity_score": 3.4,
        "human_likeness_score": 2.6,
        "average": True,
        "persona": "Average across personas",
    },
}
```
