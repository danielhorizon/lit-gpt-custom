import os
import json
import torch
from pathlib import Path
from typing import List, Union
from litgpt.tokenizer import Tokenizer

def prepare_text_dataset(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    tokenizer: Tokenizer,
    block_size: int = 512,
    chunk_size: int = 1000,
    train_val_split: float = 0.9,
    seed: int = 42,
):
    """Preprocess text files into the format required by StreamingDataset.
    
    Args:
        input_dir: Directory containing input text files
        output_dir: Directory to save processed data
        tokenizer: Tokenizer to use for encoding text
        block_size: Maximum sequence length
        chunk_size: Number of sequences per chunk file
        train_val_split: Fraction of data to use for training
        seed: Random seed for train/val split
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    train_dir = output_dir / "train" / "chunks"
    val_dir = output_dir / "val" / "chunks"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all text files
    text_files = list(input_dir.glob("*.txt"))
    if not text_files:
        raise ValueError(f"No .txt files found in {input_dir}")
    
    # Process files and create chunks
    all_sequences = []
    for text_file in text_files:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
            # Split into sentences or paragraphs (you may want to customize this)
            sequences = text.split("\n\n")
            for seq in sequences:
                if not seq.strip():
                    continue
                # Tokenize and truncate to block_size
                tokens = tokenizer.encode(seq.strip(), max_length=block_size)
                if len(tokens) < 2:  # Skip very short sequences
                    continue
                # Convert to tensor and ensure it's the right shape
                tokens = torch.tensor(tokens, dtype=torch.long)
                all_sequences.append(tokens)
    
    # Shuffle sequences
    torch.manual_seed(seed)
    indices = torch.randperm(len(all_sequences))
    all_sequences = [all_sequences[i] for i in indices]
    
    # Split into train/val
    split_idx = int(len(all_sequences) * train_val_split)
    train_sequences = all_sequences[:split_idx]
    val_sequences = all_sequences[split_idx:]
    
    # Save chunks
    def save_chunks(sequences: List[torch.Tensor], chunk_dir: Path, prefix: str):
        for i in range(0, len(sequences), chunk_size):
            chunk = sequences[i:i + chunk_size]
            chunk_path = chunk_dir / f"{prefix}_{i//chunk_size}.pt"
            # Save as a list of tensors
            torch.save(chunk, chunk_path)
    
    save_chunks(train_sequences, train_dir, "chunk")
    save_chunks(val_sequences, val_dir, "chunk")
    
    # Create index.json files
    def create_index_json(dir_path: Path, num_chunks: int):
        index = {
            "config": {
                "item_loader": "TokensLoader",
                "block_size": block_size
            },
            "chunks": [f"chunks/chunk_{i}.pt" for i in range(num_chunks)]
        }
        with open(dir_path / "index.json", "w") as f:
            json.dump(index, f, indent=2)
    
    create_index_json(output_dir / "train", len(train_sequences) // chunk_size + 1)
    create_index_json(output_dir / "val", len(val_sequences) // chunk_size + 1)
    
    print(f"Processed {len(all_sequences)} sequences")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    print(f"Output saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input text files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer", help="Directory containing tokenizer files")
    parser.add_argument("--block_size", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of sequences per chunk file")
    parser.add_argument("--train_val_split", type=float, default=0.9, help="Fraction of data to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()
    
    tokenizer = Tokenizer(args.tokenizer_dir)
    prepare_text_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        block_size=args.block_size,
        chunk_size=args.chunk_size,
        train_val_split=args.train_val_split,
        seed=args.seed,
    ) 