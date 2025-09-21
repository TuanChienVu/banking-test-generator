#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate Training Script V2 - Kaggle Compatible
Tested and optimized for Kaggle T4 GPU environment
"""

import os
import sys
import json
import gc
import time
import warnings
warnings.filterwarnings('ignore')

# Set environment variables BEFORE imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Import transformers components individually to avoid conflicts
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed
)

from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check environment
ON_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed for reproducibility
set_seed(42)

print(f"Environment: {'Kaggle' if ON_KAGGLE else 'Local'}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


@dataclass
class TrainingConfig:
    """Configuration optimized for Kaggle environment"""
    
    # Model
    model_name: str = "Salesforce/codet5-base"
    
    # Data settings - Conservative for stability
    max_input_length: int = 180  # Slightly reduced for memory
    max_target_length: int = 180
    
    # Training hyperparameters - Optimized for quality
    batch_size: int = 6  # Reduced for stability on T4
    gradient_accumulation_steps: int = 4  # Effective batch = 24
    learning_rate: float = 2e-5
    num_epochs: int = 8  # Sufficient for convergence
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Optimization
    fp16: bool = torch.cuda.is_available()  # Use mixed precision if GPU available
    gradient_checkpointing: bool = False  # Disable for stability
    
    # Evaluation & Saving
    eval_steps: int = 250
    save_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"  # Simple metric for stability
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Generation parameters
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    
    # Paths
    output_dir: str = "/kaggle/working/codet5_model" if ON_KAGGLE else "./codet5_model"
    logging_dir: str = "/kaggle/working/logs" if ON_KAGGLE else "./logs"
    
    # Data paths
    train_data: str = "datasets/augmented/train.json"
    val_data: str = "datasets/augmented/val.json"
    test_data: str = "datasets/augmented/test.json"
    
    seed: int = 42


class TestCaseDataset(Dataset):
    """Simple and robust dataset class"""
    
    def __init__(self, data_path: str, tokenizer, config: TrainingConfig):
        """Initialize dataset"""
        self.tokenizer = tokenizer
        self.config = config
        
        # Load data with error handling
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        if idx >= len(self.data):
            idx = idx % len(self.data)  # Wraparound for safety
        
        sample = self.data[idx]
        
        # Get input and target text
        input_text = str(sample.get('input_text', ''))
        target_text = str(sample.get('target_text', sample.get('output_text', '')))
        
        # Ensure non-empty
        if not input_text:
            input_text = "Generate test case"
        if not target_text:
            target_text = "Scenario: Test\n  Given system\n  When action\n  Then result"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                target_text,
                max_length=self.config.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Replace padding token id's of the labels by -100
        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }


def compute_metrics(eval_preds):
    """Simple metrics computation"""
    predictions, labels = eval_preds
    
    # Handle predictions format
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Simple loss-based metric (trainer handles this automatically)
    # We'll let the trainer compute loss as our main metric
    return {}


def evaluate_gherkin_quality(predictions, tokenizer):
    """Evaluate Gherkin structure in predictions"""
    metrics = {
        'has_scenario': 0,
        'has_given': 0,
        'has_when': 0,
        'has_then': 0,
        'complete_structure': 0
    }
    
    # Decode predictions
    if hasattr(predictions, 'predictions'):
        pred_ids = predictions.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)
        
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        
        for pred in decoded_preds:
            if 'Scenario:' in pred:
                metrics['has_scenario'] += 1
            if 'Given' in pred:
                metrics['has_given'] += 1
            if 'When' in pred:
                metrics['has_when'] += 1
            if 'Then' in pred:
                metrics['has_then'] += 1
            
            # Check complete structure
            if all(keyword in pred for keyword in ['Scenario:', 'Given', 'When', 'Then']):
                metrics['complete_structure'] += 1
        
        # Convert to percentages
        n = len(decoded_preds) if decoded_preds else 1
        for key in metrics:
            metrics[key] = metrics[key] / n
    
    return metrics


class SimpleCallback(TrainerCallback):
    """Simple callback for monitoring training"""
    
    def __init__(self):
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "="*60)
        print("🚀 TRAINING STARTED")
        print("="*60)
        print(f"Total epochs: {args.num_train_epochs}")
        print(f"Total training steps: {state.max_steps}")
        print(f"Batch size: {args.per_device_train_batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print("="*60 + "\n")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        print(f"\n📊 Epoch {int(epoch)}/{int(args.num_train_epochs)} completed")
        if self.start_time:
            elapsed = (time.time() - self.start_time) / 60
            print(f"⏱️ Time elapsed: {elapsed:.1f} minutes")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n📈 Evaluation at step {state.global_step}:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED")
        if self.start_time:
            total_time = (time.time() - self.start_time) / 60
            print(f"⏱️ Total training time: {total_time:.1f} minutes")
        print("="*60)


def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """Main training function"""
    
    # Initialize configuration
    config = TrainingConfig()
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("🔧 INITIALIZING TRAINING")
    print("="*60)
    
    # Clear memory before starting
    clear_memory()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✅ Tokenizer loaded: {config.model_name}")
    
    # Load model
    print("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    
    # Move model to device
    model = model.to(device)
    print(f"✅ Model loaded and moved to {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = TestCaseDataset(config.train_data, tokenizer, config)
    val_dataset = TestCaseDataset(config.val_data, tokenizer, config)
    test_dataset = TestCaseDataset(config.test_data, tokenizer, config)
    
    print(f"✅ Datasets loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        
        # Training
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Optimization
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        
        # Evaluation - FIXED for new transformers version
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=config.eval_steps,
        
        # Saving
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        
        # Logging
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        report_to=[],  # Disable wandb/tensorboard for simplicity
        
        # Performance
        fp16=config.fp16,
        dataloader_num_workers=0,  # Important for Kaggle
        
        # Other
        seed=config.seed,
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=None,  # Let trainer compute loss automatically
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold
            ),
            SimpleCallback()
        ]
    )
    
    # Start training
    print("\n🚀 Starting training...")
    print("="*60)
    
    try:
        # Train
        train_result = trainer.train()
        
        # Save final model
        print("\n💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        
        # Save training results
        with open(f"{config.output_dir}/training_results.json", 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        
        # Save test results
        with open(f"{config.output_dir}/test_results.json", 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        # Print final metrics
        print("\n" + "="*60)
        print("📈 FINAL METRICS")
        print("="*60)
        print(f"Training loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Test loss: {test_metrics.get('test_loss', 'N/A'):.4f}")
        
        # Test generation quality
        print("\n🔮 Testing generation quality...")
        test_input = "QA test case for mobile banking login feature"
        
        inputs = tokenizer(test_input, return_tensors='pt', max_length=config.max_input_length, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config.max_target_length,
                num_beams=config.num_beams,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                length_penalty=config.length_penalty,
                early_stopping=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nInput: {test_input}")
        print(f"Generated: {generated_text[:200]}...")
        
        # Check quality
        has_structure = all(keyword in generated_text for keyword in ['Given', 'When', 'Then'])
        print(f"\nHas Gherkin structure: {'✅' if has_structure else '❌'}")
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"Model saved to: {config.output_dir}")
        
        # Create archive if on Kaggle
        if ON_KAGGLE:
            import zipfile
            archive_path = "/kaggle/working/model_final.zip"
            print(f"\n📦 Creating archive: {archive_path}")
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(config.output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=os.path.dirname(config.output_dir))
                        zf.write(file_path, arcname)
            
            print(f"✅ Archive created: {archive_path}")
            print("📥 Ready for download from Output tab")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Clear memory
        clear_memory()


if __name__ == "__main__":
    main()
