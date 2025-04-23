import warnings
warnings.filterwarnings("ignore")

import os
import glob
import textwrap
import time
import traceback
import torch
import guidance
from guidance import models, gen, user, system, assistant
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv, find_dotenv


n = "\n"

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

        if backend_type == "transformers":
            self._init_transformers_backend(model_id, load_in_8bit, device_map, cache_dir)
        elif backend_type == "openai":
            load_dotenv(find_dotenv()) # load openai api key from ./.env
            self._init_openai_backend(model_id, openai_base_url)
        elif backend_type == "llamacpp":
            print(f"llamacpp_model_path: {llamacpp_model_path}")
            self._init_llamacpp_backend(llamacpp_model_path, llamacpp_n_ctx)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

    def _init_transformers_backend(self, model_id, load_in_8bit, device_map, cache_dir):
        """Initialize Hugging Face Transformers backend."""
        if self.verbose:
            print(f"Loading {model_id} {'with 8-bit quantization' if load_in_8bit else ''}")

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_use_double_quant=False,
        ) if load_in_8bit else None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.guidance_model = models.Transformers(
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=self.max_input_len,
            echo=False
        )

    def _init_openai_backend(self, model_id, base_url='https://api.openai.com/v1/'):
        """Initialize OpenAI backend."""
        self.guidance_model = models.OpenAI(model_id, base_url=base_url)

    def _init_llamacpp_backend(self, model_path, n_ctx):
        """Initialize Llama.cpp backend."""
        if not model_path:
            raise ValueError("Model path is required")
        from llama_cpp import Llama
        self.guidance_model = models.LlamaCpp(
                model=Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,
                    n_ctx=n_ctx,
                    verbose=False
                ),
                echo=False
            )