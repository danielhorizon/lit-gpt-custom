#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

from litgpt.finetune.roberta import RobertaFinetuner
from litgpt.data.mlm import load_mlm_dataset, create_dataloader

def main():
    parser = argparse.ArgumentParser(description="Finetune RoBERTa for MLM")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="out/roberta_finetuned", help="Output directory for the finetuned model")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Name of the pretrained RoBERTa model")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Probability of masking tokens")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize finetuner
    finetuner = RobertaFinetuner(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
    )

    # Load dataset
    train_examples = load_mlm_dataset(args.data_dir, "train")
    eval_examples = load_mlm_dataset(args.data_dir, "test")

    # Create dataloaders
    train_dataloader = create_dataloader(
        train_examples,
        finetuner.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        shuffle=True,
    )
    eval_dataloader = create_dataloader(
        eval_examples,
        finetuner.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        shuffle=False,
    )

    # Train the model
    finetuner.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        output_dir=args.output_dir,
    )

    # Final evaluation
    final_metrics = finetuner.evaluate(eval_dataloader)
    print("\nFinal evaluation metrics:")
    for metric_name, value in final_metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main() 