import os
import json
import requests
from pathlib import Path

def download_file(url: str, output_path: Path):
    """Download a file from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    output_path.write_text(response.text)

def main():
    # Create output directory
    output_dir = Path("tokenizer")
    output_dir.mkdir(exist_ok=True)
    
    # Download vocabulary
    vocab_url = "https://huggingface.co/roberta-base/raw/main/vocab.json"
    vocab_path = output_dir / "vocab.json"
    print(f"Downloading vocabulary from {vocab_url}...")
    download_file(vocab_url, vocab_path)
    
    # Download merges
    merges_url = "https://huggingface.co/roberta-base/raw/main/merges.txt"
    merges_path = output_dir / "merges.txt"
    print(f"Downloading merges from {merges_url}...")
    download_file(merges_url, merges_path)
    
    # Update tokenizer.json with vocabulary
    tokenizer_path = Path("tokenizer.json")
    with open(tokenizer_path) as f:
        tokenizer_config = json.load(f)
    
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    # Update vocabulary in tokenizer config
    tokenizer_config["model"]["vocab"] = vocab
    
    # Read merges
    with open(merges_path) as f:
        merges = [line.strip() for line in f if line.strip()]
    
    # Update merges in tokenizer config
    tokenizer_config["model"]["merges"] = merges
    
    # Save updated tokenizer config
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print("Tokenizer configuration updated successfully!")

if __name__ == "__main__":
    main() 