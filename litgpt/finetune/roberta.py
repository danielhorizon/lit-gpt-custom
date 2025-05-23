import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import math

from litgpt.utils import check_valid_checkpoint_dir, get_default_supported_precision

class RobertaForMLM(nn.Module):
    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.lm_head = nn.Linear(self.roberta.config.hidden_size, self.roberta.config.vocab_size, bias=False)
        self.lm_head.weight = self.roberta.embeddings.word_embeddings.weight  # Tie weights
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only compute loss on masked tokens
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.roberta.config.vocab_size), labels.view(-1))
            loss = masked_lm_loss
            
        return {"loss": loss, "logits": prediction_scores} if loss is not None else prediction_scores

class RobertaFinetuner:
    def __init__(
        self,
        model_name: str = "roberta-base",
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 3,
        max_length: int = 128,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_length = max_length
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMLM(model_name).to(self.device)
        
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        output_dir: str = "out/roberta_finetuned",
    ):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        best_eval_loss = float("inf")
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Average training loss: {avg_train_loss:.4f}")
            
            if eval_dataloader is not None:
                eval_metrics = self.evaluate(eval_dataloader)
                print(f"Evaluation metrics: {eval_metrics}")
                
                if eval_metrics["loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["loss"]
                    self.save_model(output_dir)
        
        if eval_dataloader is None:
            self.save_model(output_dir)
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_masked_tokens = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                total_loss += loss.item()
                
                # Calculate masked token accuracy
                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)
                
                # Only consider masked tokens (where labels != -100)
                mask = (labels != -100)
                total_masked_tokens += mask.sum().item()
                correct_predictions += ((predictions == labels) & mask).sum().item()
        
        metrics = {
            "loss": total_loss / len(eval_dataloader),
            "perplexity": math.exp(total_loss / len(eval_dataloader)),
            "masked_token_accuracy": correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0,
        }
        
        return metrics
    
    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.roberta.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.model.lm_head.state_dict(), os.path.join(output_dir, "lm_head.pt"))
    
    def load_model(self, model_dir: str):
        self.model.roberta = RobertaModel.from_pretrained(model_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        self.model.lm_head.load_state_dict(
            torch.load(os.path.join(model_dir, "lm_head.pt"))
        )
        self.model.to(self.device) 