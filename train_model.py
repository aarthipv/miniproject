#!/usr/bin/env python3
"""
Latin Text Reconstruction Model Training Script
Based on the notebook: latin_gen (2).ipynb
"""

import json
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer
)
import evaluate
import torch

def load_dataset(file_path):
    """Load the damaged Latin dataset from JSONL file."""
    print(f"Loading dataset from: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    dataset = Dataset.from_list(data)
    print(f"Dataset loaded with {len(dataset)} examples")
    print(f"Sample data: {dataset[0]}")
    return dataset

def preprocess_data(dataset, tokenizer, max_input_length=128, max_target_length=128):
    """Preprocess the dataset for training."""
    print("Preprocessing dataset...")
    
    def preprocess(example):
        model_input = tokenizer(
            example["damaged_text"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["ground_truth"],
                max_length=max_target_length,
                truncation=True,
                padding="max_length"
            )
        
        model_input["labels"] = labels["input_ids"]
        return model_input
    
    tokenized_dataset = dataset.map(preprocess, batched=True)
    print("Dataset preprocessing completed")
    return tokenized_dataset

def compute_metrics(eval_pred, tokenizer):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # BLEU score
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    
    # Accuracy (Exact Match)
    exact_matches = sum([1 for pred, label in zip(decoded_preds, decoded_labels) if pred.strip() == label.strip()])
    acc_score = exact_matches / len(decoded_preds)
    
    return {"bleu": bleu_score["bleu"], "accuracy": acc_score}

def train_model():
    """Main training function."""
    print("Starting Latin Text Reconstruction Model Training")
    print("=" * 50)
    
    # Configuration
    model_name = "facebook/bart-base"
    dataset_path = "damaged_latin_dataset (1).jsonl"
    output_dir = "./reconstruction_model"
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Load tokenizer and model
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Preprocess dataset
    tokenized_dataset = preprocess_data(dataset, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,  # Reduced batch size
        per_device_eval_batch_size=4,   # Reduced batch size
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        no_cuda=True,  # Force CPU training
        dataloader_pin_memory=False,  # Disable pin memory
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer)
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate the model
    print("Evaluating model...")
    results = trainer.evaluate()
    print(f"Final evaluation results: {results}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train_model() 