import os
import time
import traceback
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Dict
from guidance import models

class BaseDatasetProcessor(ABC):
    """Base class for dataset processing with guidance models"""

    STOP_STRINGS = ['<|im_end|>','</|im_end|>','</|im_start|>', '<|im_start|>', '```']
    
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
        name = self.config.get('save_name')
        if name is not None:
            return os.path.join(base, name)
        else:
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