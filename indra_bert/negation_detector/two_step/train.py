"""
Training script for two-step negation detection model.
Trains both cue detection and scope prediction jointly.
"""
from datetime import datetime
import argparse
from pathlib import Path
import numpy as np
from datasets import Dataset, load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer, AutoConfig, Trainer, TrainingArguments
)
from functools import partial
from transformers import EvalPrediction
import torch
import json

from .preprocess import (
    load_negation_dataset,
    build_cue_label_mapping,
    tokenize_cue_dataset,
    tokenize_scope_dataset,
)
from .model import NegationDetectorModel
from transformers import DataCollatorForTokenClassification
from typing import Any, Dict, List


class TwoStepDataCollator:
    """Data collator for two-step training."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Check if this is a cue dataset or scope dataset
        if "cue_labels" in features[0]:
            # Cue detection batch
            cue_labels = [f.pop("cue_labels") for f in features]
            batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            
            # Pad cue labels
            max_length = batch["input_ids"].shape[1]
            cue_labels_padded = []
            for labels in cue_labels:
                if len(labels) < max_length:
                    labels = labels + [-100] * (max_length - len(labels))
                else:
                    labels = labels[:max_length]
                cue_labels_padded.append(labels)
            
            batch["cue_labels"] = torch.tensor(cue_labels_padded, dtype=torch.long)
            return batch
        
        elif "scope_start_position" in features[0]:
            # Scope prediction batch
            cue_positions = [f.pop("cue_position") for f in features]
            scope_start_positions = [f.pop("scope_start_position") for f in features]
            scope_end_positions = [f.pop("scope_end_position") for f in features]
            # Remove text field (metadata, not needed for model)
            [f.pop("text", None) for f in features]
            
            batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            batch["cue_positions"] = torch.tensor(cue_positions, dtype=torch.long)
            batch["scope_start_positions"] = torch.tensor(scope_start_positions, dtype=torch.long)
            batch["scope_end_positions"] = torch.tensor(scope_end_positions, dtype=torch.long)
            return batch
        
        else:
            # Fallback: just pad
            return self.tokenizer.pad(features, padding=True, return_tensors="pt")


def compute_metrics_cue(eval_preds: EvalPrediction, cue_id2label: Dict[int, str]):
    """Compute metrics for cue detection."""
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids
    
    cue_preds = np.argmax(predictions, axis=-1)  # [N, L]
    
    TP, FP, FN = 0, 0, 0
    
    for i in range(len(cue_preds)):
        pred_ids = cue_preds[i]
        gold_ids = labels[i]
        
        # Extract spans from predictions
        pred_spans = []
        current_span = None
        for j, label_id in enumerate(pred_ids):
            if label_id == -100:
                continue
            label = cue_id2label.get(label_id, "O")
            if label == "B-cue":
                if current_span:
                    pred_spans.append(current_span)
                current_span = (j, j)
            elif label == "I-cue" and current_span:
                current_span = (current_span[0], j)
            elif label == "O":
                if current_span:
                    pred_spans.append(current_span)
                    current_span = None
        
        if current_span:
            pred_spans.append(current_span)
        
        # Extract spans from gold labels
        gold_spans = []
        current_span = None
        for j, label_id in enumerate(gold_ids):
            if label_id == -100:
                continue
            label = cue_id2label.get(label_id, "O")
            if label == "B-cue":
                if current_span:
                    gold_spans.append(current_span)
                current_span = (j, j)
            elif label == "I-cue" and current_span:
                current_span = (current_span[0], j)
            elif label == "O":
                if current_span:
                    gold_spans.append(current_span)
                    current_span = None
        
        if current_span:
            gold_spans.append(current_span)
        
        pred_set = set(pred_spans)
        gold_set = set(gold_spans)
        
        TP += len(pred_set & gold_set)
        FP += len(pred_set - gold_set)
        FN += len(gold_set - pred_set)
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "cue_precision": precision,
        "cue_recall": recall,
        "cue_f1": f1,
        "cue_TP": TP,
        "cue_FP": FP,
        "cue_FN": FN,
    }


def compute_metrics_scope(eval_preds: EvalPrediction) -> Dict[str, Any]:
    """
    Compute span-level precision / recall / F1 for scope prediction.
    Assumes:
      - eval_preds.predictions is (start_logits, end_logits)
      - eval_preds.label_ids is (gold_start_positions, gold_end_positions)
    """
    # Unpack predictions
    # For a QA-style model, Trainer passes a tuple
    predictions = eval_preds.predictions
    if isinstance(predictions, tuple) and len(predictions) == 2:
        start_logits, end_logits = predictions
    else:
        # Fallback: treat as [N, L, 2] and split
        start_logits = predictions[..., 0]
        end_logits = predictions[..., 1]

    # Argmax over sequence dimension -> predicted indices
    pred_start = np.argmax(start_logits, axis=-1)  # [N]
    pred_end = np.argmax(end_logits, axis=-1)      # [N]

    # Unpack labels
    labels = eval_preds.label_ids
    if isinstance(labels, tuple) and len(labels) == 2:
        gold_start, gold_end = labels
    else:
        # Fallback: labels shaped [N, 2]
        gold_start = labels[:, 0]
        gold_end = labels[:, 1]

    # Convert to int and compute span-based TP/FP/FN
    TP, FP, FN = 0, 0, 0
    num_examples = len(pred_start)

    for i in range(num_examples):
        gs, ge = int(gold_start[i]), int(gold_end[i])
        ps, pe = int(pred_start[i]), int(pred_end[i])

        # If you ever use -100 for missing labels, skip such examples
        if gs < 0 or ge < 0:
            continue

        gold_span = (gs, ge)
        pred_span = (ps, pe)

        if pred_span == gold_span:
            TP += 1
        else:
            # one gold span per example, one predicted span per example
            FN += 1  # missed gold correctly
            FP += 1  # predicted wrong span

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    start_acc = float((pred_start == gold_start).sum()) / num_examples
    end_acc   = float((pred_end == gold_end).sum()) / num_examples

    return {
        "scope_precision": precision,
        "scope_recall": recall,
        "scope_f1": f1,
        "scope_TP": TP,
        "scope_FP": FP,
        "scope_FN": FN,
        "scope_start_accuracy": start_acc,
        "scope_end_accuracy": end_acc,
    }


class ScopeTrainer(Trainer):
    """Custom trainer for scope prediction that handles scope labels."""
    
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Custom compute_loss that pulls scope labels from inputs and passes
        them to the model.
        """
        # Pull label tensors out of inputs (added by DataCollator)
        scope_start_positions = inputs.pop("scope_start_positions", None)
        scope_end_positions = inputs.pop("scope_end_positions", None)
        cue_positions = inputs.pop("cue_positions", None)

        if scope_start_positions is None or scope_end_positions is None:
            # Helpful debugging; remove once stable
            print("ERROR: Missing scope positions. Inputs keys:", list(inputs.keys()))
            raise ValueError(
                "scope_start_positions and scope_end_positions must be provided in inputs"
            )

        # Forward pass: your NegationDetectorModel.forward should accept these kwargs
        outputs = model(
            **inputs,
            scope_start_positions=scope_start_positions,
            scope_end_positions=scope_end_positions,
            cue_positions=cue_positions,
        )

        # Standard HF pattern
        loss = outputs.loss if hasattr(outputs, "loss") else None
        if loss is None:
            raise ValueError(
                "Model did not return a loss. "
                "Make sure your model returns loss when given scope_*_positions."
            )

        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-step negation detection model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to combined_negation_dataset.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", help="Pretrained model name or path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for datasets")
    parser.add_argument("--use_cached_dataset", action="store_true", help="Use cached dataset if available")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--train_jointly", action="store_true", help="Train both stages jointly (alternating batches)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dataset_path = output_dir / "cached_dataset"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load and split dataset
    if args.use_cached_dataset and cache_dataset_path.exists():
        print("Loading from cache...")
        cached = load_from_disk(cache_dataset_path)
        cue_train_dataset = cached["cue_train"]
        cue_val_dataset = cached["cue_val"]
        cue_test_dataset = cached["cue_test"]
        scope_train_dataset = cached["scope_train"]
        scope_val_dataset = cached["scope_val"]
        scope_test_dataset = cached["scope_test"]
        
        with open(cache_dataset_path / "cue_label2id.json", "r") as f:
            cue_label2id = json.load(f)
    else:
        print("Loading and preprocessing dataset...")
        examples = load_negation_dataset(Path(args.data_path))
        
        # Build label mappings
        cue_label2id, cue_id2label = build_cue_label_mapping()
        
        # Split dataset
        np.random.seed(42)
        indices = np.random.permutation(len(examples))
        n_train = int(len(examples) * args.train_split)
        n_val = int(len(examples) * args.val_split)
        
        train_examples = [examples[i] for i in indices[:n_train]]
        val_examples = [examples[i] for i in indices[n_train:n_train+n_val]]
        test_examples = [examples[i] for i in indices[n_train+n_val:]]
        
        print(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
        
        # Create cue detection datasets
        cue_train_dataset = tokenize_cue_dataset(train_examples, tokenizer, cue_label2id)
        cue_val_dataset = tokenize_cue_dataset(val_examples, tokenizer, cue_label2id)
        cue_test_dataset = tokenize_cue_dataset(test_examples, tokenizer, cue_label2id)
        
        # Create scope prediction datasets
        scope_train_dataset = tokenize_scope_dataset(train_examples, tokenizer)
        scope_val_dataset = tokenize_scope_dataset(val_examples, tokenizer)
        scope_test_dataset = tokenize_scope_dataset(test_examples, tokenizer)
        
        print(f"Cue datasets - Train: {len(cue_train_dataset)}, Val: {len(cue_val_dataset)}")
        print(f"Scope datasets - Train: {len(scope_train_dataset)}, Val: {len(scope_val_dataset)}")
        
        # Save cache
        cached = DatasetDict(
            cue_train=cue_train_dataset,
            cue_val=cue_val_dataset,
            cue_test=cue_test_dataset,
            scope_train=scope_train_dataset,
            scope_val=scope_val_dataset,
            scope_test=scope_test_dataset,
        )
        cached.save_to_disk(cache_dataset_path)
        
        with open(cache_dataset_path / "cue_label2id.json", "w") as f:
            json.dump(cue_label2id, f)
    
    cue_id2label = {v: k for k, v in cue_label2id.items()}
    
    # Load model config
    config = AutoConfig.from_pretrained(args.model_name)
    config.cue_num_labels = len(cue_label2id)
    
    # Initialize model
    model = NegationDetectorModel.from_pretrained(
        args.model_name,
        config=config,
        cue_num_labels=len(cue_label2id),
    )
    
    # Resize token embeddings if needed
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        print("Resizing token embeddings...")
        model.resize_token_embeddings(len(tokenizer))
    
    
    # Data collators
    cue_collator = TwoStepDataCollator(tokenizer=tokenizer)
    scope_collator = TwoStepDataCollator(tokenizer=tokenizer)
    
    # For now, train on cue dataset first, then scope dataset
    cue_ckpt_dir = output_dir / "cue_checkpoint"
    scope_ckpt_dir = output_dir / "scope_checkpoint"
    cue_ckpt_dir.mkdir(parents=True, exist_ok=True)
    scope_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Train on cue detection
    print("\n" + "="*80)
    print("TRAINING CUE DETECTION")
    print("="*80)

    # Training arguments
    cue_training_args = TrainingArguments(
        output_dir=cue_ckpt_dir,
        eval_strategy=args.eval_strategy,
        save_strategy=args.eval_strategy,
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="cue_f1",
        greater_is_better=True,
    )
    
    cue_trainer = Trainer(
        model=model,
        args=cue_training_args,
        train_dataset=cue_train_dataset,
        eval_dataset=cue_val_dataset,
        tokenizer=tokenizer,
        data_collator=cue_collator,
        compute_metrics=partial(compute_metrics_cue, cue_id2label=cue_id2label),
    )
    
    cue_trainer.train()
    cue_trainer.save_model(cue_ckpt_dir)
    
    # Train on scope prediction (fine-tuning)
    print("\n" + "="*80)
    print("TRAINING SCOPE PREDICTION")
    print("="*80)
    
    # Update training args for scope fine-tuning
    scope_config = AutoConfig.from_pretrained(cue_ckpt_dir)

    if not hasattr(scope_config, "cue_num_labels"):
        scope_config.cue_num_labels = len(cue_label2id)

    model = NegationDetectorModel.from_pretrained(
        cue_ckpt_dir,
        config=scope_config,
    )

    scope_training_args = TrainingArguments(
        output_dir=scope_ckpt_dir,
        eval_strategy=args.eval_strategy,
        save_strategy=args.eval_strategy,
        logging_strategy="epoch",
        learning_rate=args.learning_rate * 0.5,  # Lower LR for fine-tuning
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="scope_f1",
        greater_is_better=True,
        remove_unused_columns=False,
    )
    
    scope_trainer = ScopeTrainer(
        model=model,
        args=scope_training_args,
        train_dataset=scope_train_dataset,
        eval_dataset=scope_val_dataset,
        tokenizer=tokenizer,
        data_collator=scope_collator,
        compute_metrics=compute_metrics_scope,
    )

    scope_trainer.label_names = ["scope_start_positions", "scope_end_positions"]

    scope_trainer.train()

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Evaluate on test set
    test_metrics_cue = cue_trainer.evaluate(eval_dataset=cue_test_dataset)
    test_metrics_scope = scope_trainer.evaluate(eval_dataset=scope_test_dataset)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"test_eval_results_{timestamp}.txt"
    with open(log_file, "w") as f:
        f.write("CUE DETECTION METRICS:\n")
        for key, value in test_metrics_cue.items():
            f.write(f"{key}: {value}\n")
        f.write("\nSCOPE PREDICTION METRICS:\n")
        for key, value in test_metrics_scope.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Test evaluation results saved to {log_file}")
    
    # Save final model
    print("Saving final model...")
    scope_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings
    with open(output_dir / "cue_label2id.json", "w") as f:
        json.dump(cue_label2id, f)


if __name__ == "__main__":
    main()

