import warnings
warnings.filterwarnings("ignore")

import os
import time
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv, find_dotenv


class BaseGenerator:
    """Base class for handling different LLM backends."""
    
    def __init__(
        self,
        backend_type: str = "transformers",
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        load_in_8bit: bool = False,
        max_input_len: int = 4096,
        cache_dir: str = "/data2/.shared_models",
        device_map: str = "auto",
        verbose: bool = False,
        # OpenAI-specific parameters
        openai_base_url: Optional[str] = None,
        # LlamaCpp-specific parameters
        llamacpp_model_path: Optional[str] = None,
        llamacpp_n_ctx: int = 2048,
    ):
        self.backend_type = backend_type
        self.verbose = verbose
        self.max_input_len = max_input_len
        self.model_id = model_id
        self.stop_strings = ['<|im_end|>','</|im_end|>','</|im_start|>', '<|im_start|>', '```', 
                          '<|reserved_special_token_*|>', '---', '.Human:', '.Assistant']

        if backend_type == "transformers":
            self._init_transformers_backend(model_id, load_in_8bit, device_map, cache_dir)
        elif backend_type == "openai":
            self._init_openai_backend(model_id, openai_base_url)
        elif backend_type == "llama.cpp":
            self._init_llamacpp_backend(llamacpp_model_path, llamacpp_n_ctx)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

    def _init_transformers_backend(self, model_id, load_in_8bit, device_map, cache_dir):
        """Initialize Hugging Face Transformers backend directly."""
        if self.verbose:
            print(f"Loading {model_id} {'with 8-bit quantization' if load_in_8bit else ''} using Transformers")
        
        # Set the device first
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set up appropriate configurations based on hardware and parameters
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        if load_in_8bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        else:
            quantization_config = None
        
        # Safer model loading with better error handling
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.verbose:
                    print(f"Setting pad_token to eos_token ({self.tokenizer.eos_token})")
            
            # Load model with appropriate parameters
            model_kwargs = {
                "cache_dir": cache_dir,
                "trust_remote_code": True,
            }

            
            # Only add GPU specific configurations if on CUDA
            if self.device == "cuda":
                model_kwargs.update({
                    "device_map": device_map,
                    "torch_dtype": torch_dtype,
                })
                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config
            
            if "gemma" in model_id.lower():
                model_kwargs["attn_implementation"] = "eager"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )

            # if "gemma" in model_id.lower():
            #     self.model.config.sliding_window = None

            tok_vocab   = len(self.tokenizer)
            model_vocab = self.model.get_input_embeddings().num_embeddings
            if tok_vocab != model_vocab:
                if self.verbose:
                    print(f"[warn] tokenizer vocab ({tok_vocab}) "
                        f"≠ model vocab ({model_vocab}); resizing model.")
            self.model.resize_token_embeddings(tok_vocab)
            
            if self.device == "cpu" and self.verbose:
                print("Model loaded on CPU - performance will be limited")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise
            
        # Check if it's an instruction-tuned model that needs special formatting
        self.is_llama = "llama" in model_id.lower()
        self.is_mistral = "mistral" in model_id.lower()

    def _init_openai_backend(self, model_id, base_url=None):
        """Initialize OpenAI backend."""
        from openai import OpenAI
        
        load_dotenv(find_dotenv())  # load openai api key from ./.env
        
        self.client = OpenAI(base_url=base_url)
        self.model_id = model_id

    def _init_llamacpp_backend(self, model_path, n_ctx):
        """Initialize Llama.cpp backend."""
        if not model_path:
            raise ValueError("Model path is required")
        from llama_cpp import Llama
        
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            verbose=False
        )
    
    def _format_prompt(self, system_prompt, user_prompt):
        """Format the prompt based on model type."""
        if self.backend_type == "transformers":
            if self.is_llama:
                # Llama style formatting
                if system_prompt:
                    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                else:
                    full_prompt = ""
                full_prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                return full_prompt
            elif self.is_mistral:
                # Mistral style formatting
                if system_prompt:
                    full_prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
                else:
                    full_prompt = f"<s>[INST] {user_prompt} [/INST]"
                return full_prompt
            else:
                # Generic instruction tuned model
                if system_prompt:
                    return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant: "
                else:
                    return f"User: {user_prompt}\n\nAssistant: "
        elif self.backend_type == "openai":
            # For OpenAI, we'll handle this formatting in the generate method
            return user_prompt
        elif self.backend_type == "llama.cpp":
            # Similar to generic transformers format
            if system_prompt:
                return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant: "
            else:
                return f"User: {user_prompt}\n\nAssistant: "
    
    def generate(self, 
                 user_prompt: str, 
                 system_prompt: Optional[str] = None,
                 max_tokens: int = 1000,
                 temperature: float = 1.0,
                 top_p: float = 0.95,
                 top_k: int = 50,
                 do_sample: bool = True) -> Optional[Dict[str, Any]]:
        """Generate text based on the given prompt."""
        try:
            start_time = time.time()
            
            if self.backend_type == "transformers":
                formatted_prompt = self._format_prompt(system_prompt, user_prompt)
                
                # Tokenize the prompt
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, 
                                     max_length=self.max_input_len)
                
                # Make sure we're using the right device
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                if (inputs["input_ids"] >= self.model.get_input_embeddings().num_embeddings).any():
                    raise ValueError("Some token ids exceed the model’s embedding size; check tokenizer/model pair.")
                
                # Generate text with safer parameters
                with torch.no_grad():
                    # For some models, pad_token_id might not be properly set
                    if self.tokenizer.pad_token_id is None:
                        pad_token_id = self.tokenizer.eos_token_id
                    else:
                        pad_token_id = self.tokenizer.pad_token_id
                    
                    generation_config = {
                        "max_new_tokens": max_tokens,
                        "pad_token_id": pad_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                    }
                    
                    # Only add sampling parameters if do_sample is True
                    if do_sample:
                        generation_config.update({
                            "do_sample": True,
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                        })
                    
                    # if hasattr(self.model.config, "sliding_window") and self.model.config.sliding_window:
                    #     win = self.model.config.sliding_window
                    #     if inputs["input_ids"].shape[-1] > win:
                    #         inputs = {k: v[:, -win:] for k, v in inputs.items()}
                    
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                
                # Decode the generated text
                full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the generated part by removing the prompt
                if full_output.startswith(formatted_prompt):
                    generated_text = full_output[len(formatted_prompt):]
                else:
                    # If we can't find the exact prompt, try to extract the assistant's response
                    # by looking for common patterns in instruction-following formats
                    if "Assistant:" in full_output:
                        generated_text = full_output.split("Assistant:", 1)[1].strip()
                    else:
                        generated_text = full_output
                
                # Handle stop strings
                for stop_str in self.stop_strings:
                    if stop_str in generated_text:
                        generated_text = generated_text.split(stop_str)[0]
                
            elif self.backend_type == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature if do_sample else 0.0,
                    top_p=top_p if do_sample else 1.0
                )
                
                generated_text = response.choices[0].message.content
            
            elif self.backend_type == "llama.cpp":
                formatted_prompt = self._format_prompt(system_prompt, user_prompt)
                
                response = self.model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature if do_sample else 0.0,
                    top_p=top_p if do_sample else 1.0,
                    stop=self.stop_strings,
                    echo=False
                )
                
                generated_text = response["choices"][0]["text"]
            
            return {
                "text": generated_text.strip(),
                "generation_time": time.time() - start_time
            }
            
        except Exception as e:
            print(f"Generation error: {e}")
            traceback.print_exc()
            return None