import os
import logging
import math
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader
from litdata.streaming import StreamingDataset, StreamingDataLoader, TokensLoader

from litgpt.model import Roberta
from litgpt.config import Config
from litgpt.data.wikitext import WikiTextMLMDataModule
from litgpt.tokenizer import Tokenizer
from litgpt.utils import choose_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobertaLightningModule(L.LightningModule):
    def __init__(
        self,
        model: Roberta,
        config: dict,
    ):
        super().__init__()
        self.model = model
        self.config = config
        
    def forward(self, input_ids):
        return self.model(input_ids)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        logits = self(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # Create parameter groups for weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["optimizer"]["init_args"]["weight_decay"],
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Get optimizer class and arguments from config
        optimizer_class = self.config["optimizer"]["class_path"]
        optimizer_args = self.config["optimizer"]["init_args"]
        for key, value in optimizer_args.items():
            if isinstance(value, str):
                try:
                    optimizer_args[key] = float(value)
                except ValueError:
                    pass  # Keep non-numeric strings as is
        
        # Import optimizer class dynamically
        module_path, class_name = optimizer_class.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        optimizer_class = getattr(module, class_name)
        
        # Create optimizer
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_args)
        
        # Get scheduler class and arguments from config
        scheduler_class = self.config["lr_scheduler"]["class_path"]
        scheduler_args = self.config["lr_scheduler"]["init_args"]
        
        # Import scheduler class dynamically
        module_path, class_name = scheduler_class.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        scheduler_class = getattr(module, class_name)
        
        # Calculate total training steps
        num_training_steps = self.trainer.estimated_stepping_batches
        warmup_steps = scheduler_args.pop("warmup_steps", 0)
        
        # Create scheduler
        scheduler_args["lr_lambda"] = lambda step: self.lr_lambda(step, warmup_steps, num_training_steps)
        scheduler = scheduler_class(optimizer, **scheduler_args)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def lr_lambda(self, current_step, warmup_steps, num_training_steps):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))

class StreamingMLMDataModule:
    """Streaming data module for MLM pretraining."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Tokenizer,
        block_size: int = 512,
        micro_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.micro_batch_size = micro_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Setup paths
        self.train_path = self.data_path / "train"
        self.val_path = self.data_path / "val"
        
    def prepare_data(self):
        """Prepare the data for streaming."""
        if not self.train_path.exists() or not self.val_path.exists():
            raise FileNotFoundError(
                f"Data paths not found. Expected {self.train_path} and {self.val_path} to exist."
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create streaming train dataloader."""
        dataset = StreamingDataset(
            input_dir=str(self.train_path),
            item_loader=TokensLoader(block_size=self.block_size),
            shuffle=True,
            seed=self.seed,
        )
        return StreamingDataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create streaming validation dataloader."""
        dataset = StreamingDataset(
            input_dir=str(self.val_path),
            item_loader=TokensLoader(block_size=self.block_size),
            shuffle=False,
        )
        return StreamingDataLoader(
            dataset,
            batch_size=self.val_batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

def get_data_module(config: dict, tokenizer: Tokenizer, test_mode: bool = False):
    """Get the appropriate data module based on config and test mode."""
    if test_mode:
        logger.info("Using WikiTextMLMDataModule for testing")
        return WikiTextMLMDataModule(
            tokenizer=tokenizer,
            block_size=config["data"]["block_size"],
            train_batch_size=config["data"]["train_batch_size"],
            val_batch_size=config["data"]["val_batch_size"],
        )
    
    logger.info("Using StreamingMLMDataModule for training")
    return StreamingMLMDataModule(
        data_path=config["data"]["data_path"],
        tokenizer=tokenizer,
        block_size=config["data"]["block_size"],
        micro_batch_size=config["data"]["micro_batch_size"],
        val_batch_size=config["data"]["val_batch_size"],
        num_workers=config["data"].get("num_workers", 4),
        seed=config["data"].get("seed", 42),
    )

def main(config_path: str = "config_hub/pretrain/roberta-base.yaml", test_mode: bool = False):
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info("Starting RoBERTa pretraining")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(config.get("tokenizer_dir", "tokenizer"))
    logger.info("Tokenizer loaded")
    
    # Create output directory
    out_dir = Path("out/roberta_wikitext" if test_mode else config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")
    
    # Initialize model
    model_config = Config(
        n_layer=12,
        n_embd=768,
        vocab_size=50265,  # RoBERTa's vocab size
        block_size=512,
        bias=True,
    )
    model = Roberta(model_config)
    logger.info("Model initialized")
    
    # Initialize data module
    data_module = get_data_module(config, tokenizer, test_mode)
    
    # Initialize lightning module
    lightning_module = RobertaLightningModule(
        model=model,
        config=config,
    )
    
    # Setup logging and checkpointing
    tb_logger = TensorBoardLogger(out_dir, name="logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        filename="roberta-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
    )
    
    # Initialize trainer with distributed training settings
    trainer = L.Trainer(
        max_epochs=config["train"]["epochs"],
        devices=config["train"].get("devices", "auto"),
        accelerator="auto",
        strategy="ddp",
        precision=config["train"].get("precision", "32-true"),
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=config["train"].get("gradient_clip_val", 1.0),
        accumulate_grad_batches=config["train"].get("gradient_accumulation_steps", 1),
    )
    
    # Train
    trainer.fit(lightning_module, data_module)
    
    # Save final model
    if trainer.is_global_zero:
        final_model_path = out_dir / "roberta_final.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_hub/pretrain/roberta-base.yaml")
    parser.add_argument("--test_mode", action="store_true", help="Use WikiText dataset for testing")
    args = parser.parse_args()
    main(args.config, args.test_mode) 