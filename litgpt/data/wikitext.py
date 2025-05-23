import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


class WikiTextMLMDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer: Tokenizer,
        block_size: int = 512,
        mask_prob: float = 0.15,
        mask_token_id: int = 50264,  # Roberta's mask token
        pad_token_id: int = 0,  # Roberta's pad token
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        
        # Load WikiText-2 dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        self.examples = self._process_dataset(dataset)
        
    def _process_dataset(self, dataset) -> List[torch.Tensor]:
        """Process the dataset into tokenized examples."""
        examples = []
        current_example = []
        
        for item in dataset:
            if item["text"].strip():  # Skip empty lines
                # Tokenize the text using LitGPT's tokenizer
                tokens = self.tokenizer.encode(
                    item["text"].strip(),
                    bos=False,  # Don't add BOS token for MLM
                    eos=False,  # Don't add EOS token for MLM
                    max_length=self.block_size
                )
                current_example.extend(tokens.tolist())
                
                # If we've accumulated enough tokens, create an example
                while len(current_example) >= self.block_size:
                    example = torch.tensor(current_example[:self.block_size], dtype=torch.long)
                    examples.append(example)
                    current_example = current_example[self.block_size:]
        
        # Add the last example if it's not empty
        if current_example:
            example = torch.tensor(current_example[:self.block_size], dtype=torch.long)
            if len(example) == self.block_size:  # Only add if it's full length
                examples.append(example)
                
        return examples
    
    def _create_masked_input(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create masked input and labels for MLM."""
        input_ids = tokens.clone()
        labels = tokens.clone()
        
        # Create a mask for tokens that can be masked (excluding special tokens)
        maskable_tokens = (tokens != self.tokenizer.bos_id) & \
                         (tokens != self.tokenizer.eos_id) & \
                         (tokens != self.pad_token_id)
        
        # Randomly select tokens to mask
        mask = torch.rand(tokens.shape) < self.mask_prob
        mask = mask & maskable_tokens
        
        # Replace masked tokens with [MASK] token
        input_ids[mask] = self.mask_token_id
        
        # Set labels to -100 for non-masked tokens (they will be ignored in loss computation)
        labels[~mask] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.examples[idx]
        return self._create_masked_input(tokens)


class WikiTextMLMDataModule(DataModule):
    def __init__(
        self,
        tokenizer: Tokenizer,
        block_size: int = 512,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        self.train_dataset = WikiTextMLMDataset(
            "train",
            self.tokenizer,
            block_size=self.block_size
        )
        self.val_dataset = WikiTextMLMDataset(
            "validation",
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