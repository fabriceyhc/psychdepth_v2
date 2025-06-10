import os
import time
import traceback
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Dict
from guidance import models
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class BaseDatasetProcessor(ABC):
    """Base class for dataset processing with guidance models"""

    STOP_STRINGS = ['<|im_end|>','</|im_end|>','</|im_start|>', '<|im_start|>', '```', '<tool_call>', '\<|endoftext|>', '<|end|>', '<|endoftext|>']
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = self._init_model()
        self.dataset = self.load_dataset()
        self.examples = self.load_examples() if self.config.get('examples_path') else None
        
    @abstractmethod
    def load_dataset(self):
        """Load main dataset to process"""
        pass
    
    def load_examples(self):
        """Load few-shot examples if needed"""
        examples = pd.read_csv(self.config.get('examples_path'))
        return examples
    
    @abstractmethod
    def process_entry(self, row: Dict) -> Dict:
        """Process a single dataset entry"""
        pass

    @abstractmethod
    def grade_answer(self, predicted_answer, ground_truth) -> bool:
        """Grade the predicted answer"""
        pass
    
    def _init_model(self):
        """Initialize guidance model with proper cache configuration"""
        model_cfg = self.config['model']
        
        if model_cfg['type'] == 'transformers':
            return models.Transformers(
                model_cfg['path'],
                device_map="auto",
                cache_dir=model_cfg.get('cache_dir', "./data2/.shared_models"),
                echo=False
            )
        elif model_cfg['type'] == 'llama.cpp':
            return models.LlamaCpp(
                model=model_cfg['path'],
                echo=False,
                n_gpu_layers=-1,
                n_ctx=model_cfg.get('context_size', 4096))
    
    def run(self):
        """Main processing loop"""
        results = []
        save_path = self._get_save_path()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            existing = pd.read_csv(save_path)
        except FileNotFoundError:
            existing = pd.DataFrame()

        for idx, row in tqdm(self.dataset.iterrows(), total=len(self.dataset)):
            if self._is_processed(row, existing):
                print(f"Already processed row with id={idx}. Skipping!")
                continue
                
            try:
                result = self.process_entry(row)
                results.append(result)
                self._save_incremental(results, existing, save_path)
            except Exception as e:
                self._handle_error(row, e)

        return pd.concat([existing, pd.DataFrame(results)])
    
    def _get_save_path(self) -> str:
        """Generate save path from config"""
        base = self.config['save_dir']
        model_name = os.path.basename(self.config['model']['path']).replace('.gguf', '')
        return f"{base}/{model_name}_{self.config['shots']}shot.csv"
    
    @abstractmethod
    def _is_processed(self, row: Dict, existing: pd.DataFrame) -> bool:
        """Check if row already exists in saved results"""
        pass
    
    def _save_incremental(self, results: List, existing: pd.DataFrame, path: str):
        """Save results incrementally"""
        pd.concat([existing, pd.DataFrame(results)]).to_csv(path, index=False)
    
    def _handle_error(self, row: Dict, error: Exception):
        """Error logging and handling"""
        print(f"Error processing row: {row.get('doc_id', 'unknown')}")
        print(traceback.format_exc())
        if self.config.get('error_log'):
            with open(self.config['error_log'], 'a') as f:
                f.write(f"{time.time()}\t{str(error)}\n")

# class BaseDatasetProcessor(ABC):
#     """Base class for dataset processing with transformers models"""

#     STOP_STRINGS = ['\n\n', 'Human:', 'User:', '<|im_end|>', '</|im_end|>', '</|im_start|>', '<|im_start|>', '```']
    
#     def __init__(self, config: Dict):
#         self.config = config
#         self.model_config = config.get("model", {})
#         self.save_dir = config.get("save_dir", "./results")
#         self.shots = config.get("shots", 0)
#         self.examples = self.load_examples() if config.get('examples_path') else None
        
#         # Initialize model and tokenizer
#         self._initialize_model()
#         # Load dataset after model initialization
#         self.dataset = self.load_dataset()
    
#     def _initialize_model(self):
#         """Initialize the model and tokenizer based on configuration"""
#         model_path = self.model_config.get("path", "meta-llama/Llama-3.2-1B-Instruct")
#         cache_dir = self.model_config.get("cache_dir", None)
        
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path, 
#             cache_dir=cache_dir,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
    
#     def generate_text(self, prompt, max_new_tokens=1024, stop_strings=None):
#         """Generate text using the model"""
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
#         # Generate with stopping criteria
#         generated_ids = self.model.generate(
#             inputs.input_ids,
#             max_new_tokens=max_new_tokens,
#             pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
#             attention_mask=inputs.attention_mask,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95,
#             temperature=0.5
#         )
        
#         # Decode the generated text
#         generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
#         # Remove the prompt from the generated text
#         response = generated_text[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        
#         # Apply stop strings if provided
#         if stop_strings:
#             for stop_string in stop_strings:
#                 if stop_string in response:
#                     response = response.split(stop_string)[0]
        
#         return response.strip()
    
#     @abstractmethod
#     def load_dataset(self):
#         """Load main dataset to process"""
#         pass
    
#     def load_examples(self):
#         """Load few-shot examples if needed"""
#         examples = pd.read_csv(self.config.get('examples_path'))
#         return examples

#     @abstractmethod
#     def process_entry(self, row: Dict) -> Dict:
#         """Process a single dataset entry"""
#         pass

#     @abstractmethod
#     def grade_answer(self, predicted_answer, ground_truth) -> bool:
#         """Grade the predicted answer"""
#         pass
    
#     def run(self):
#         """Main processing loop"""
#         results = []
#         save_path = self._get_save_path()
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
#         try:
#             existing = pd.read_csv(save_path)
#         except FileNotFoundError:
#             existing = pd.DataFrame()

#         for idx, row in tqdm(self.dataset.iterrows(), total=len(self.dataset)):
#             if self._is_processed(row, existing):
#                 print(f"Already processed row with id={row.get('unique_id', idx)}. Skipping!")
#                 continue
                
#             try:
#                 result = self.process_entry(row)
#                 results.append(result)
#                 self._save_incremental(results, existing, save_path)
#                 print(f"Correct: {result.get('is_correct', False)}")
#             except Exception as e:
#                 self._handle_error(row, e)

#         return pd.concat([existing, pd.DataFrame(results)]) if not existing.empty else pd.DataFrame(results)
    
#     def _get_save_path(self) -> str:
#         """Generate save path from config"""
#         base = self.save_dir
#         model_name = os.path.basename(self.model_config.get('path', '')).replace('.gguf', '')
#         return f"{base}/{model_name}_{self.shots}shot.csv"
    
#     @abstractmethod
#     def _is_processed(self, row: Dict, existing: pd.DataFrame) -> bool:
#         """Check if row already exists in saved results"""
#         pass
    
#     def _save_incremental(self, results: List, existing: pd.DataFrame, path: str):
#         """Save results incrementally"""
#         pd.concat([existing, pd.DataFrame(results)]).to_csv(path, index=False)
    
#     def _handle_error(self, row: Dict, error: Exception):
#         """Error logging and handling"""
#         print(f"Error processing row: {row.get('unique_id', 'unknown')}")
#         print(traceback.format_exc())
#         if self.config.get('error_log'):
#             with open(self.config['error_log'], 'a') as f:
#                 f.write(f"{time.time()}\t{str(error)}\n")