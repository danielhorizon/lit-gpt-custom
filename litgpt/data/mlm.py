"""Dataset utilities for Masked Language Modeling (MLM) with RoBERTa."""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling (MLM) with RoBERTa.
    
    This dataset handles text data for MLM pretraining, where tokens are randomly masked
    and the model is trained to predict the original tokens.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        mlm_probability: float = 0.15,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Create input_ids and attention_mask
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Create labels (same as input_ids for MLM)
        labels = input_ids.clone()
        
        # Create random mask for MLM
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # Replace masked tokens with [MASK] token
        input_ids[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def load_mlm_dataset(data_dir: str, split: str = "train") -> List[str]:
    """Load MLM dataset from a directory.
    
    Args:
        data_dir: Directory containing the dataset files
        split: Dataset split to load ("train" or "test")
        
    Returns:
        List of text examples
    """
    data_dir = Path(data_dir)
    file_path = data_dir / f"{split}.txt"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    return texts

def create_dataloader(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 128,
    mlm_probability: float = 0.15,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for MLM training or evaluation.
    
    Args:
        texts: List of text examples
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        mlm_probability: Probability of masking tokens
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader for the dataset
    """
    dataset = MLMDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        mlm_probability=mlm_probability,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Disable multiprocessing for simplicity
    ) 