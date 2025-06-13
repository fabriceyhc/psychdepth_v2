import warnings
warnings.filterwarnings("ignore")

import os
import glob
import textwrap
import time
import traceback
import guidance
from guidance import models, gen, user, system, assistant
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Dict, Any, List

from dataset.strategies.base import BaseGenerator

n = "\n"
STOP_STRINGS = ['<|im_end|>','</|im_end|>','</|im_start|>', '<|im_start|>', '```', '<|reserved_special_token_*|>', '---']

@guidance
def story_task(lm, premise, num_words, profile, examples, temperature):
    if profile:
        with system():
            lm += profile
    with user():
        lm += f"Please write a {num_words}-word story on:{n}{premise}{n}Only respond with the story text."
        if examples:
            lm += f"{n}### Examples:"
            for ex in examples:
                lm += f"{n}Premise: {ex['premise']}{n}Story: {ex['story_excerpt']}"
    with assistant():
        lm += gen(name="story", max_tokens=int(num_words*2), temperature=temperature, stop=STOP_STRINGS, do_sample=True)
    return lm

class WriterProfileGenerator(BaseGenerator):
    """Specialized generator for creating stories with different writing profiles."""
    
    def __init__(
        self,
        # Base parameters
        backend_type: str = "transformers",
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        load_in_8bit: bool = False,
        max_input_len: int = 4096,
        cache_dir: str = "/data2/.shared_models",
        device_map: str = "auto",
        verbose: bool = False,
        # OpenAI-specific
        openai_base_url: Optional[str] = None,
        # LlamaCpp-specific
        llamacpp_model_path: Optional[str] = None,
        llamacpp_n_ctx: int = 2048,
    ):
        super().__init__(
            backend_type=backend_type,
            model_id=model_id,
            load_in_8bit=load_in_8bit,
            max_input_len=max_input_len,
            cache_dir=cache_dir,
            device_map=device_map,
            verbose=verbose,
            openai_base_url=openai_base_url,
            llamacpp_model_path=llamacpp_model_path,
            llamacpp_n_ctx=llamacpp_n_ctx,
        )

        self.default_profile = textwrap.dedent("""\
            You are a seasoned writer who has won several accolades for your emotionally rich stories. 
            When you write, you delve deep into the human psyche, pulling from the reservoir of universal 
            experiences that every reader, regardless of their background, can connect to.
            Your writing is renowned for painting vivid emotional landscapes, making readers not just observe 
            but truly feel the world of your characters.
            Every piece you produce aims to draw readers in, encouraging them to reflect on their own lives 
            and emotions.
            Your stories are a complex tapestry of relationships, emotions, and conflicts, 
            each more intricate than the last.
            """
        )

    def generate_story(
        self, 
        premise: str, 
        num_words: int = 500, 
        profile: Optional[str] = None,
        examples: Optional[List[Dict]] = None,
        temperature: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """Generate a story with the given premise and writing profile."""
        try:
            start_time = time.time()
            output = self.guidance_model + story_task(
                premise=premise,
                num_words=num_words,
                profile=profile or self.default_profile,
                examples=examples or [],
                temperature=temperature
            )
            return {
                "story": output["story"].strip(),
                "generation_time": time.time() - start_time
            }
        except Exception as e:
            print(f"Generation error: {e}")
            traceback.print_exc()
            return None


# Example usage remains the same as original
if __name__ == "__main__":
    generator = WriterProfileGenerator(model_id="meta-llama/Llama-3.2-1B-Instruct")
    premise = "A world where politicians receive downvotes instead of votes..."
    result = generator.generate_story(premise=premise, num_words=50)
    print("Generated Story:\n", result["story"])