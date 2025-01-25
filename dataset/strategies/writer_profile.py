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

n = "\n"



class WriterProfileGenerator:
    def __init__(
        self,
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        load_in_8bit=False,
        max_input_len=4096,
        cache_dir="/data2/.shared_models",
        device_map="auto",
        verbose=False
    ):
        """
        Initialize with bitsandbytes 8-bit quantization
        """
        if verbose:
            print(f"Loading {model_id} {'with 8-bit quantization' if load_in_8bit else ''}")

        if load_in_8bit:
            # Configure 8-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_use_double_quant=False,
            )

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config if load_in_8bit else None,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True,
            use_cache=True

        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )

        self.max_input_len = max_input_len
        self.verbose = verbose

        # Set a default writer profile (system prompt)
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

    def generate_story(self, 
                       premise: str, 
                       num_words: int = 500, 
                       profile: str = None,
                       examples: list = None,
                       temperature: float = 1.0):
        """
        Generate a story with the given premise and writer profile. 
        - `premise`: The main idea or scenario for the story.
        - `num_words`: Roughly how many words you want in the story.
        - `profile`: An optional 'system' message describing the writer's style/persona.
        - `examples`: An optional list of few-shot examples (each a dict), if you want to provide them.
        """

        @guidance(dedent=True)
        def story_task(lm, premise, num_words, profile, examples, temperature):
            # If a writer profile is provided, treat it like a system message.
            if profile:
                with system():
                    lm += f"{profile}"

            # The user instruction (requesting a story).
            with user():
                lm += f"""
                Please write a {num_words}-word story on the following prompt:
                
                {premise}
                
                Only respond with the story text.
                """
                # (Optional) If you want to embed example-based prompting:
                if examples and len(examples) > 0:
                    lm += "### Examples:\n"
                    for ex in examples:
                        lm += f"{n}Example premise: {ex['premise']}\n"
                        lm += f"Example story excerpt: {ex['story_excerpt']}\n"

            # The assistant block—where we capture the actual story generation.
            with assistant():
                lm += gen(name="story", max_tokens=int(num_words*2), temperature=temperature)
                
            return lm

        # If user didn't provide a profile, use the default
        if not profile:
            profile = self.default_profile

        # Execute the Guidance program with your model
        start_time = time.time()
        try:

            # Reconstructing guidance interface to reset state
            guidance_model = guidance.models.Transformers(
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.max_input_len,
                echo=False
            )

            output = guidance_model + story_task(
                premise=premise,
                num_words=num_words,
                profile=profile or "",
                examples=examples or [],
                temperature=temperature
            )
            generation_time = time.time() - start_time

            return {
                "story": output["story"].strip(),
                "generation_time": generation_time
            }

        except Exception as e:
            print(f"An error occurred during generation: {e}")
            traceback.print_exc()
            return None


if __name__ == "__main__":

    # TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=7 python -m dataset.strategies.writer_profile

    # Example usage
    generator = WriterProfileGenerator(model_id="meta-llama/Llama-3.2-1B-Instruct")

    # If you provide a custom profile, it overrides the default
    custom_profile = textwrap.dedent("""\
        You are a futuristic sci-fi writer specializing in post-apocalyptic worlds and robot protagonists.
        All your stories reflect high-tech environments, philosophical themes, and machine-human interactions.
        """
    )

    # A simple premise
    premise = "You are allowed to 'downvote' a government candidate instead of voting normally, reducing their votes by one. Turns out people have little love for politicians, and the majority end with negative votes. In these democracies, anonymity is the key to winning."

    # (Optional) A few-shot example
    few_shot_examples = [
        {
            "premise": "A fisherman who hears voices under the sea.",
            "story_excerpt": "He sank his hands into the brine, recalling the echoes of a silent past..."
        }
    ]

    result = generator.generate_story(
        premise=premise, 
        num_words=50, 
        # profile=custom_profile,
        # examples=few_shot_examples
    )

    print("Generated Story:\n", result["story"])
    print("Time Taken (sec):", result["generation_time"])
