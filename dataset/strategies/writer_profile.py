import textwrap
import time
import traceback
from typing import Optional, Dict, Any, List
import re

from dataset.strategies.base import BaseGenerator

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

    def _format_story_prompt(self, premise: str, num_words: int, examples: Optional[List[Dict]] = None) -> str:
        """Format the prompt for story generation."""
        newline = "\n"
        user_prompt = f"Please write a {num_words}-word story on:{newline}{premise}{newline}Only respond with the story text."
        
        if examples and len(examples) > 0:
            user_prompt += f"{newline}### Examples:"
            for ex in examples:
                user_prompt += f"{newline}Premise: {ex['premise']}{newline}Story: {ex['story_excerpt']}"
        
        return user_prompt

    def generate_story(
        self, 
        premise: str, 
        num_words: int = 500, 
        profile: Optional[str] = None,
        examples: Optional[List[Dict]] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> Optional[Dict[str, Any]]:
        """Generate a story with the given premise and writing profile."""
        try:
            start_time = time.time()
            
            # Set the system prompt to the writing profile
            system_prompt = profile or self.default_profile
            
            # Format the user prompt
            user_prompt = self._format_story_prompt(premise, num_words, examples)
            
            # Generate the story
            result = self.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=int(num_words * 2) + (2000 if 'Qwen3' in self.model_id else 0),  # Allocate enough tokens for the story
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )
            
            if not result:
                return None
            
            # Clean up the generated text to ensure we only get the story part
            story_text = result["text"].strip()
            
            # Remove potential leftover instructions or formatting
            # Remove common prefixes that might appear in the generated text
            prefixes_to_remove = [
                "Here's a story:", "Story:", "Here is a story:", 
                "Here is the story:", "My story:"
            ]
            
            for prefix in prefixes_to_remove:
                if story_text.startswith(prefix):
                    story_text = story_text[len(prefix):].strip()
            
            # Clean up potential markdown formatting
            if story_text.startswith("```") and story_text.endswith("```"):
                story_text = re.sub(r"```.*?\n", "", story_text, 1)  # Remove opening markdown
                story_text = re.sub(r"\n```$", "", story_text)  # Remove closing markdown
            
            return {
                "story": story_text.strip(),
                "generation_time": time.time() - start_time
            }
            
        except Exception as e:
            print(f"Story generation error: {e}")
            traceback.print_exc()
            return None


# Example usage
if __name__ == "__main__":
    generator = WriterProfileGenerator(model_id="meta-llama/Llama-3.2-1B-Instruct")
    premise = "A world where politicians receive downvotes instead of votes..."
    result = generator.generate_story(premise=premise, num_words=50)
    print("Generated Story:\n", result["story"])