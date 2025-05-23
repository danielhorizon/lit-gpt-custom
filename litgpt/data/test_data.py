import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import torch
from lightning.fabric.utilities.throughput import ThroughputMonitor
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Literal

from litgpt import Tokenizer
from litgpt.args import TrainArgs
from litgpt.data import DataModule


class TestMLMDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        tokenizer: Tokenizer,
        block_size: int = 512,
        mask_prob: float = 0.15,
        mask_token_id: int = 50264,  # Roberta's mask token
        pad_token_id: int = 0,  # Roberta's pad token
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        
        # Create test data if it doesn't exist
        test_data_file = self.data_dir / "test_data.txt"
        if not test_data_file.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._create_test_data()
            
        self.examples = self._load_examples()
        
    def _create_test_data(self):
        """Create a small test dataset with example sentences."""
        example_sentences = [
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing helps computers understand human language.",
            "Deep learning models can learn complex patterns from data.",
            "Transformers have revolutionized natural language processing.",
            "BERT and RoBERTa are popular transformer-based models.",
            "Masked language modeling is a self-supervised learning task.",
            "Pretraining helps models learn general language understanding.",
            "Fine-tuning adapts pretrained models to specific tasks.",
            "Attention mechanisms help models focus on relevant information.",
            "Tokenization converts text into numerical representations."
        ]
        
        with open(self.data_dir / "test_data.txt", "w") as f:
            f.write("\n".join(example_sentences))
            
    def _load_examples(self) -> List[torch.Tensor]:
        """Load and tokenize the test data."""
        examples = []
        with open(self.data_dir / "test_data.txt", "r") as f:
            for line in f:
                if line.strip():
                    # Tokenize the sentence
                    tokens = self.tokenizer.encode(line.strip())
                    # Convert to tensor
                    tokens = torch.tensor(tokens, dtype=torch.long)
                    examples.append(tokens)
        return examples
    
    def _create_masked_input(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create masked input and labels for MLM."""
        input_ids = tokens.clone()
        labels = tokens.clone()
        
        # Create a mask for tokens that can be masked (excluding special tokens)
        maskable_tokens = (tokens != self.tokenizer.cls_token_id) & \
                         (tokens != self.tokenizer.sep_token_id) & \
                         (tokens != self.tokenizer.pad_token_id)
        
        # Randomly select tokens to mask
        mask = torch.rand(tokens.shape) < self.mask_prob
        mask = mask & maskable_tokens
        
        # Replace masked tokens with [MASK] token
        input_ids[mask] = self.mask_token_id
        
        # Set labels to -100 for non-masked tokens (they will be ignored in loss computation)
        labels[~mask] = -100
        
        return input_ids, labels
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.examples[idx]
        
        # Truncate or pad to block_size
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        else:
            tokens = torch.cat([tokens, torch.full((self.block_size - len(tokens),), self.pad_token_id)])
            
        # Create masked input and labels
        input_ids, labels = self._create_masked_input(tokens)
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }


class TestMLMDataModule(DataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        tokenizer: Tokenizer,
        block_size: int = 512,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        self.train_dataset = TestMLMDataset(
            self.data_dir,
            self.tokenizer,
            block_size=self.block_size
        )
        self.val_dataset = TestMLMDataset(
            self.data_dir,
            self.tokenizer,
            block_size=self.block_size
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        ) 