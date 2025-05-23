# Roberta-base Implementation in Lit-GPT

This repository now includes support for the Roberta-base architecture, allowing you to perform masked language modeling (MLM) pretraining. The implementation follows HuggingFace-style conventions while integrating seamlessly with the existing Lit-GPT codebase.

## Features

- Roberta-base model architecture with MLM support
- Compatible with existing training/inference loops
- Test dataset for MLM pretraining
- Modular design for easy extension

## Model Architecture

The Roberta-base implementation includes:

- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- 3072 intermediate size
- 50,265 vocabulary size
- 512 maximum sequence length

## Usage

### Pretraining

To pretrain a Roberta-base model on the test dataset:

```bash
python -m litgpt pretrain \
    --model_name roberta-base \
    --data_dir path/to/test_data \
    --out_dir path/to/output \
    --train.global_batch_size 32 \
    --train.micro_batch_size 4 \
    --train.max_tokens 1000000 \
    --train.learning_rate 1e-4 \
    --train.lr_warmup_steps 2000 \
    --train.save_interval 1000 \
    --eval.interval 1000
```

### Test Dataset

The test dataset is automatically created if it doesn't exist. It includes a small set of example sentences for testing the MLM pretraining pipeline. The dataset is created in the specified data directory with the following structure:

```
data_dir/
    test.txt  # Contains example sentences for MLM pretraining
```

### Custom Dataset

To use your own dataset for MLM pretraining:

1. Create a new dataset class inheriting from `TestMLMDataset`
2. Implement the required methods:
   - `_load_examples()`: Load and tokenize your data
   - `_create_masked_input()`: Create masked input and labels
   - `__getitem__()`: Return input_ids, labels, and attention_mask

Example:

```python
from litgpt.data.test_data import TestMLMDataset

class CustomMLMDataset(TestMLMDataset):
    def _load_examples(self):
        # Load your custom data here
        examples = []
        with open(self.data_dir / "your_data.txt", "r") as f:
            for line in f:
                tokens = self.tokenizer.encode(line.strip(), max_length=self.block_size)
                if len(tokens) < 2:
                    continue
                examples.append(torch.tensor(tokens, dtype=torch.long))
        return examples
```

## Extending the Implementation

The implementation is designed to be modular and extensible:

1. **Tokenizer Training**: Add your own tokenizer training pipeline by extending the existing tokenizer classes
2. **Inference**: Use the model for inference by loading a pretrained checkpoint
3. **Finetuning**: Extend the finetuning pipeline to support Roberta-specific tasks

## Model Checkpoints

The pretrained model checkpoints are saved in the output directory with the following structure:

```
out_dir/
    iter-{step}.pt  # Regular checkpoints
    best.pt         # Best checkpoint based on validation loss
```

## Notes

- The implementation uses Roberta's special tokens (mask token ID: 50264, pad token ID: 1)
- The default masking probability is 15% (configurable)
- The model supports both MLM pretraining and regular language modeling
- The implementation is compatible with the existing Lit-GPT training infrastructure

# RoBERTa Finetuning

This document provides instructions for finetuning RoBERTa models on text classification tasks.

## Installation

Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Dataset Format

The finetuning script supports two types of datasets:

1. Toy Dataset (default): A simple binary classification dataset for movie reviews
2. AG News Dataset: A 4-class news classification dataset

### Toy Dataset Format

The toy dataset should be organized as follows:

```
data/
  test_data_finetune/
    train.json
    test.json
```

Each JSON file should contain a list of examples, where each example has the following format:

```json
{
    "text": "The movie was fantastic!",
    "label": 1
}
```

### AG News Dataset Format

The AG News dataset should be organized similarly, with each line in the JSON files containing a news article and its category label.

## Finetuning

To finetune RoBERTa on the toy dataset:

```bash
python scripts/finetune_roberta.py \
    --data_dir data/test_data_finetune \
    --output_dir out/roberta_finetuned \
    --num_labels 2 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3
```

To finetune on the AG News dataset:

```bash
python scripts/finetune_roberta.py \
    --data_dir path/to/agnews \
    --output_dir out/roberta_finetuned \
    --num_labels 4 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --use_agnews
```

## Command Line Arguments

- `--data_dir`: Directory containing the dataset (required)
- `--output_dir`: Output directory for the finetuned model (default: "out/roberta_finetuned")
- `--model_name`: Name of the pretrained RoBERTa model (default: "roberta-base")
- `--num_labels`: Number of classification labels (default: 2)
- `--batch_size`: Training batch size (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--num_epochs`: Number of training epochs (default: 3)
- `--max_length`: Maximum sequence length (default: 128)
- `--use_agnews`: Use AG News dataset instead of toy dataset (flag)

## Evaluation

The script automatically evaluates the model on the test set after training. The following metrics are computed:

- Loss
- Accuracy
- F1 Score
- Precision
- Recall

The final metrics are printed to the console after training completes.

## Model Output

The finetuned model is saved in the specified output directory with the following files:

- `config.json`: Model configuration
- `pytorch_model.bin`: Model weights
- `tokenizer.json`: Tokenizer configuration
- `classifier.pt`: Classification head weights

## Example

Here's a complete example of finetuning on the toy dataset:

```bash
# Create data directory
mkdir -p data/test_data_finetune

# Copy the toy dataset files
cp data/test_data_finetune/train.json data/test_data_finetune/
cp data/test_data_finetune/test.json data/test_data_finetune/

# Run finetuning
python scripts/finetune_roberta.py \
    --data_dir data/test_data_finetune \
    --output_dir out/roberta_finetuned \
    --num_labels 2 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3
```

The script will train the model and output evaluation metrics. The finetuned model will be saved in the `out/roberta_finetuned` directory. 