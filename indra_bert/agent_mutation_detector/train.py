from datetime import datetime
import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification, 
                          AutoConfig, Trainer, TrainingArguments)
from functools import partial
from transformers import EvalPrediction

from .preprocess import (
    preprocess_for_training,
    preprocess_for_inference
)

from transformers import DataCollatorForTokenClassification
from typing import Any, Dict, List
import torch

# ---- Data Collator ----
class DataCollatorWithDebug(DataCollatorForTokenClassification):
    def __init__(self, tokenizer, id2label, max_examples_to_print=1, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.counter = 0
        self.max_examples_to_print = max_examples_to_print

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features)

        if self.counter < self.max_examples_to_print:
            for i in range(min(len(features), self.max_examples_to_print - self.counter)):
                input_ids = batch["input_ids"][i]
                labels = batch["labels"][i]
                attention_mask = batch["attention_mask"][i]

                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                label_names = [self.id2label.get(l.item(), "IGN") if l.item() != -100 else "PAD" for l in labels]

                print("\n--- DEBUG: Mutation Training Example ---")
                print("-" * 50)
                for j, tok in enumerate(tokens):
                    attn = attention_mask[j].item()
                    label = label_names[j]
                    print(f"{tok:15} {label:12} {attn:<9}")
                print("-" * 50)

            self.counter += len(features)

        return batch

# ---- Metrics ----
def compute_metrics_span_level(eval_preds: EvalPrediction, inputs, id2label, texts):
    """
    Compute span-level metrics for mutation detection.
    """
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids

    TP, FP, FN = 0, 0, 0

    for i in range(len(predictions)):
        pred_ids = np.argmax(predictions[i], axis=1).tolist()
        gold_ids = labels[i].tolist()

        tokens = inputs[i]["tokens"]
        text = texts[i]
        
        # Create dummy offsets since we don't have them in the input
        # This is a simplified approach - in practice you'd want to store offsets
        offsets = [(0, len(token)) for token in tokens]

        # Extract mutation spans from predictions and labels
        pred_spans = extract_mutation_spans(tokens, offsets, pred_ids, id2label, text)
        gold_spans = extract_mutation_spans(tokens, offsets, gold_ids, id2label, text)

        pred_set = {(s["start"], s["end"], s["text"]) for s in pred_spans}
        gold_set = {(s["start"], s["end"], s["text"]) for s in gold_spans}

        TP += len(pred_set & gold_set)
        FP += len(pred_set - gold_set)
        FN += len(gold_set - pred_set)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}

def extract_mutation_spans(tokens, offset_mapping, predictions, id2label, original_text):
    """
    Extract mutation spans from token-level predictions.
    """
    mutation_spans = []
    current_span = None
    
    for i, (token, offset, pred_id) in enumerate(zip(tokens, offset_mapping, predictions)):
        if offset[0] == offset[1]:  # Skip special tokens
            continue
            
        label = id2label.get(pred_id, "O")
        
        if label == "B-mutation":
            # Start new mutation span
            if current_span:
                mutation_spans.append(current_span)
            current_span = {
                "start": offset[0],
                "end": offset[1],
                "text": original_text[offset[0]:offset[1]]
            }
        elif label == "I-mutation" and current_span:
            # Continue current mutation span
            current_span["end"] = offset[1]
            current_span["text"] = original_text[current_span["start"]:offset[1]]
        else:
            # End current span if exists
            if current_span:
                mutation_spans.append(current_span)
                current_span = None
    
    # Add final span if exists
    if current_span:
        mutation_spans.append(current_span)
    
    return mutation_spans

def parse_args():
    parser = argparse.ArgumentParser(description="Train Agent Mutation Detector")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSONL")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model")
    parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", help="Base model name")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--version", type=str, default="1.0", help="Model version")
    parser.add_argument("--use_cached_dataset", action="store_true", help="Use cached dataset if available")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<e>', '</e>']})
    
    # Preprocess data
    print("Preprocessing data...")
    all_examples, label2id, id2label = preprocess_for_training(
        args.dataset_path, tokenizer, max_length=512
    )
    
    # Split into train/val (80/20 split)
    split_idx = int(0.8 * len(all_examples))
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Labels: {label2id}")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    # Create model config
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        config=config
    )
    
    # Resize token embeddings to match tokenizer size
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        print("Resizing token embeddings to match tokenizer size...")
        print("Old embedding size:", model.get_input_embeddings().num_embeddings)
        print("New embedding size:", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))
    
    # Create data collator
    data_collator = DataCollatorWithDebug(
        tokenizer=tokenizer,
        id2label=id2label,
        max_examples_to_print=2
    )
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir='./logs',
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=partial(
            compute_metrics_span_level,
            inputs=val_examples,
            id2label=id2label,
            texts=[ex["text"] for ex in val_examples]
        ),
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save label mappings
    import json
    with open(f"{args.output_dir}/label2id.json", "w") as f:
        json.dump(label2id, f)
    with open(f"{args.output_dir}/id2label.json", "w") as f:
        json.dump(id2label, f)
    
    print(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
