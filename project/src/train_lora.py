"""
Training script for LoRA finetuning using unsloth.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import torch
import hydra
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, Trainer, set_seed, AutoTokenizer
from unsloth import FastLanguageModel
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import TrainOutput

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_name_or_path: str = "unsloth/Meta-Llama-3.1-8B"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    dataset_path: str = "/afs/cs.stanford.edu/u/vvajipey/research/continual/gisting/data/alpaca_plus"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

@dataclass
class TrainingArguments(TrainingArguments):
    """Training arguments."""
    output_dir: str = "outputs"
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    seed: int = 42
    remove_unused_columns: bool = False  # Required for unsloth
    bf16: bool = True  # Use bfloat16 for training
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"

class MetricsCallback(TrainerCallback):
    """Custom callback for tracking and plotting metrics."""
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_steps': [],
            'train_steps': []
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called on each log event."""
        if logs is None:
            return
        
        # Track training loss
        if 'loss' in logs:
            self.metrics_history['train_loss'].append(logs['loss'])
            self.metrics_history['train_steps'].append(state.global_step)
        
        # Track evaluation loss
        if 'eval_loss' in logs:
            self.metrics_history['eval_loss'].append(logs['eval_loss'])
            self.metrics_history['eval_steps'].append(state.global_step)
        
        # Save metrics
        self._save_metrics()
        
        # Plot metrics
        if state.global_step % args.logging_steps == 0:
            self._plot_metrics()
    
    def _save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = self.output_dir / 'metrics_history.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f)
    
    def _plot_metrics(self):
        """Plot and save training metrics."""
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        if self.metrics_history['train_loss']:
            plt.plot(
                self.metrics_history['train_steps'],
                self.metrics_history['train_loss'],
                label='Training Loss'
            )
        
        # Plot evaluation loss
        if self.metrics_history['eval_loss']:
            plt.plot(
                self.metrics_history['eval_steps'],
                self.metrics_history['eval_loss'],
                label='Evaluation Loss',
                marker='o'
            )
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(self.output_dir / 'training_progress.png')
        plt.close()

def format_instruction(example):
    """Format instruction and response into prompt format."""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    response = example["output"]
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    
    return {
        "text": prompt,
        "input_ids": None,  # Will be filled by tokenizer
        "attention_mask": None,  # Will be filled by tokenizer
        "labels": None,  # Will be filled by tokenizer
    }

def load_alpaca_dataset(data_args: DataArguments) -> DatasetDict:
    """Load the Alpaca+ dataset."""
    def load_json(filename: str) -> List[dict]:
        with open(os.path.join(data_args.dataset_path, filename), 'r') as f:
            return json.load(f)

    # Load datasets
    train_data = load_json("alpaca_plus_train.json")
    validation_seen = load_json("alpaca_plus_validation_seen.json")
    validation_human = load_json("alpaca_plus_validation_human.json")
    validation_unseen = load_json("alpaca_plus_validation_unseen.json")

    # Convert to datasets
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation_seen": Dataset.from_list(validation_seen),
        "validation_human": Dataset.from_list(validation_human),
        "validation_unseen": Dataset.from_list(validation_unseen),
    })

    # Apply formatting
    dataset = dataset.map(
        format_instruction,
        num_proc=4,
        remove_columns=dataset["train"].column_names,  # Remove original columns
    )

    # Limit dataset size if specified
    if data_args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(data_args.max_train_samples))
    if data_args.max_eval_samples is not None:
        dataset["validation_seen"] = dataset["validation_seen"].select(range(data_args.max_eval_samples))
        dataset["validation_human"] = dataset["validation_human"].select(range(data_args.max_eval_samples))
        dataset["validation_unseen"] = dataset["validation_unseen"].select(range(data_args.max_eval_samples))

    return dataset

def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions, labels = eval_preds
    
    # Remove padding
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate metrics
    metrics = {
        "exact_match": np.mean([p.strip() == l.strip() for p, l in zip(pred_texts, label_texts)]),
    }
    
    return metrics

@hydra.main(config_path="../config", config_name="train_lora", version_base=None)
def main(args: DictConfig) -> None:
    """Main training function."""
    print(OmegaConf.to_yaml(args))
    
    # Set random seed
    set_seed(args.training.seed)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.training.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    args.training.output_dir = str(output_dir)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(args))
    
    # Load model and tokenizer
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model.model_name_or_path,
        max_seq_length=args.model.max_seq_length,
        load_in_4bit=args.model.load_in_4bit,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and process dataset
    dataset = load_alpaca_dataset(DataArguments(**args.data))
    
    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize the texts with truncation and padding
        results = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.model.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        results["labels"] = results["input_ids"].copy()
        return results
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )
    
    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.model.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.model.lora_alpha,
        lora_dropout=args.model.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.training.seed,
    )
    
    # Initialize trainer with metrics and callback
    metrics_callback = MetricsCallback(output_dir=args.training.output_dir)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**args.training),
        train_dataset=tokenized_dataset["train"],
        eval_dataset={
            "seen": tokenized_dataset["validation_seen"],
            "human": tokenized_dataset["validation_human"],
            "unseen": tokenized_dataset["validation_unseen"]
        },
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback]
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()

if __name__ == "__main__":
    main()
