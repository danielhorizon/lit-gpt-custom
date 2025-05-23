#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

from litgpt.finetune.roberta import RobertaFinetuner
from litgpt.data.mlm import load_mlm_dataset, create_dataloader

def main():
    parser = argparse.ArgumentParser(description="Evaluate a finetuned RoBERTa model on MLM")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the test dataset")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the finetuned model")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Probability of masking tokens")
    args = parser.parse_args()

    # Initialize finetuner and load the finetuned model
    finetuner = RobertaFinetuner(
        model_name=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    finetuner.load_model(args.model_dir)

    # Load test dataset
    test_examples = load_mlm_dataset(args.data_dir, "test")
    test_dataloader = create_dataloader(
        test_examples,
        finetuner.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        shuffle=False,
    )

    # Evaluate the model
    metrics = finetuner.evaluate(test_dataloader)
    print("\nEvaluation metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main() 