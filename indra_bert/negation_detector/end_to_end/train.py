"""
Training script for negation detection model.
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

from .preprocess import (
    load_negation_dataset,
    build_label_mappings,
    tokenize_negation_dataset,
    IGNORE_INDEX,
)
from .model import NegationDetectorModel
from .postprocess import extract_spans_from_encoding
from transformers import DataCollatorForTokenClassification
from typing import Any, Dict, List


def extract_spans_from_token_labels(
    label_ids: List[int],
    id2label: Dict[int, str],
    offset_mapping: List[tuple],
    text: str,
    span_type: str = "cue",
) -> List[tuple]:
    """
    Extract spans as (start, end, text) tuples from token labels.
    Used for span-level metrics.
    """
    spans = []
    current_span = None
    
    for i, (label_id, offset) in enumerate(zip(label_ids, offset_mapping)):
        if label_id == IGNORE_INDEX:
            continue
        
        start_char, end_char = offset
        if start_char is None or end_char is None:
            if current_span:
                spans.append(current_span)
                current_span = None
            continue
        
        label = id2label.get(label_id, "O")
        
        if label == "O" or not label.endswith(f"-{span_type}"):
            if current_span:
                spans.append(current_span)
                current_span = None
            continue
        
        tag = label.split("-")[0]
        
        if tag == "B":
            if current_span:
                spans.append(current_span)
            current_span = (start_char, end_char, text[start_char:end_char])
        elif tag == "I":
            if current_span:
                # Extend current span
                old_start, _, _ = current_span
                current_span = (old_start, end_char, text[old_start:end_char])
            else:
                current_span = (start_char, end_char, text[start_char:end_char])
    
    if current_span:
        spans.append(current_span)
    
    return spans


def compute_metrics_span_level(
    eval_preds: EvalPrediction,
    cue_id2label: Dict[int, str],
    scope_id2label: Dict[int, str],
    tokenizer,
    texts: List[str],
    offset_mappings: List[List[tuple]],
) -> Dict[str, float]:
    """
    Compute span-level metrics for both cues and scopes.
    """
    # Get predictions and labels
    # eval_preds.predictions contains cue_logits (from model.logits)
    # We need to get scope_logits from the trainer's stored predictions
    
    cue_logits = eval_preds.predictions  # [N, L, cue_num_labels]
    labels = eval_preds.label_ids  # [N, L] - cue labels
    
    cue_preds = np.argmax(cue_logits, axis=-1)  # [N, L]
    
    # For scope predictions, we need to get them from the trainer
    # For now, we'll compute metrics only for cues
    # TODO: Store scope_logits in trainer for metrics computation
    
    TP_cue, FP_cue, FN_cue = 0, 0, 0
    
    for i in range(len(cue_preds)):
        pred_ids = cue_preds[i]
        gold_ids = labels[i]
        
        text = texts[i]
        offset_mapping = offset_mappings[i]
        
        pred_spans = set(extract_spans_from_token_labels(
            pred_ids.tolist(), cue_id2label, offset_mapping, text, span_type="cue"
        ))
        gold_spans = set(extract_spans_from_token_labels(
            gold_ids.tolist(), cue_id2label, offset_mapping, text, span_type="cue"
        ))
        
        TP_cue += len(pred_spans & gold_spans)
        FP_cue += len(pred_spans - gold_spans)
        FN_cue += len(gold_spans - pred_spans)
    
    precision_cue = TP_cue / (TP_cue + FP_cue + 1e-8)
    recall_cue = TP_cue / (TP_cue + FN_cue + 1e-8)
    f1_cue = 2 * precision_cue * recall_cue / (precision_cue + recall_cue + 1e-8)
    
    return {
        "cue_precision": precision_cue,
        "cue_recall": recall_cue,
        "cue_f1": f1_cue,
        "cue_TP": TP_cue,
        "cue_FP": FP_cue,
        "cue_FN": FN_cue,
    }


class MultiHeadDataCollator(DataCollatorForTokenClassification):
    """Data collator that handles both cue_labels and scope_labels."""
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract labels before padding
        cue_labels = [f.pop("cue_labels") for f in features]
        scope_labels = [f.pop("scope_labels") for f in features]
        
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        
        # Pad labels to match sequence length
        max_length = batch["input_ids"].shape[1]
        cue_labels_padded = []
        scope_labels_padded = []
        
        for cue_l, scope_l in zip(cue_labels, scope_labels):
            if len(cue_l) < max_length:
                cue_l = cue_l + [IGNORE_INDEX] * (max_length - len(cue_l))
            else:
                cue_l = cue_l[:max_length]
            
            if len(scope_l) < max_length:
                scope_l = scope_l + [IGNORE_INDEX] * (max_length - len(scope_l))
            else:
                scope_l = scope_l[:max_length]
            
            cue_labels_padded.append(cue_l)
            scope_labels_padded.append(scope_l)
        
        batch["cue_labels"] = torch.tensor(cue_labels_padded, dtype=torch.long)
        batch["scope_labels"] = torch.tensor(scope_labels_padded, dtype=torch.long)
        
        return batch


class NegationTrainer(Trainer):
    """Custom trainer that stores scope_logits for metrics computation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_scope_logits = []
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to store scope_logits."""
        has_labels = any(k in inputs for k in ["cue_labels", "scope_labels"])
        
        if not prediction_loss_only:
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss if has_labels else None
                
                # Store scope_logits for metrics
                if hasattr(outputs, 'scope_logits'):
                    self._current_scope_logits.append(
                        outputs.scope_logits.detach().cpu().numpy()
                    )
                
                # Return cue_logits as primary predictions
                predictions = outputs.logits.detach()
                
                # Get labels
                labels = None
                if has_labels:
                    labels = inputs.get("cue_labels").detach()
                
                return (loss, predictions, labels)
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Clear scope_logits before evaluation."""
        self._current_scope_logits = []
        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self._current_scope_logits = []
        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train negation detection model")
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
        train_dataset = cached["train"]
        val_dataset = cached["validation"]
        test_dataset = cached["test"]
        
        # Load label mappings
        import json
        with open(cache_dataset_path / "cue_label2id.json", "r") as f:
            cue_label2id = json.load(f)
        with open(cache_dataset_path / "scope_label2id.json", "r") as f:
            scope_label2id = json.load(f)
    else:
        print("Loading and preprocessing dataset...")
        examples = load_negation_dataset(Path(args.data_path))
        
        # Build label mappings
        (
            cue_label2id, cue_id2label,
            scope_label2id, scope_id2label
        ) = build_label_mappings()
        
        # Split dataset
        np.random.seed(42)
        indices = np.random.permutation(len(examples))
        n_train = int(len(examples) * args.train_split)
        n_val = int(len(examples) * args.val_split)
        
        train_examples = [examples[i] for i in indices[:n_train]]
        val_examples = [examples[i] for i in indices[n_train:n_train+n_val]]
        test_examples = [examples[i] for i in indices[n_train+n_val:]]
        
        print(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
        
        # Tokenize datasets
        train_dataset = tokenize_negation_dataset(train_examples, tokenizer, cue_label2id, scope_label2id)
        val_dataset = tokenize_negation_dataset(val_examples, tokenizer, cue_label2id, scope_label2id)
        test_dataset = tokenize_negation_dataset(test_examples, tokenizer, cue_label2id, scope_label2id)
        
        # Save cache
        cached = DatasetDict(train=train_dataset, validation=val_dataset, test=test_dataset)
        cached.save_to_disk(cache_dataset_path)
        
        import json
        with open(cache_dataset_path / "cue_label2id.json", "w") as f:
            json.dump(cue_label2id, f)
        with open(cache_dataset_path / "scope_label2id.json", "w") as f:
            json.dump(scope_label2id, f)
    
    cue_id2label = {v: k for k, v in cue_label2id.items()}
    scope_id2label = {v: k for k, v in scope_label2id.items()}
    
    # Load model config
    config = AutoConfig.from_pretrained(args.model_name)
    config.cue_num_labels = len(cue_label2id)
    config.scope_num_labels = len(scope_label2id)
    config.cue_label2id = cue_label2id
    config.cue_id2label = cue_id2label
    config.scope_label2id = scope_label2id
    config.scope_id2label = scope_id2label
    
    # Initialize model
    model = NegationDetectorModel.from_pretrained(
        args.model_name,
        config=config,
        cue_num_labels=len(cue_label2id),
        scope_num_labels=len(scope_label2id),
    )
    
    # Resize token embeddings if needed
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        print("Resizing token embeddings...")
        model.resize_token_embeddings(len(tokenizer))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
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
    
    # Data collator
    data_collator = MultiHeadDataCollator(tokenizer=tokenizer)
    
    # Prepare texts and offset_mappings for metrics
    val_texts = [ex["text"] for ex in val_dataset]
    val_offset_mappings = []
    for ex in val_dataset:
        text = ex["text"]
        encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
        val_offset_mappings.append(encoding["offset_mapping"])
    
    # Trainer
    trainer = NegationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(
            compute_metrics_span_level,
            cue_id2label=cue_id2label,
            scope_id2label=scope_id2label,
            tokenizer=tokenizer,
            texts=val_texts,
            offset_mappings=val_offset_mappings,
        ),
    )
    
    # Train
    trainer.train()
    
    # Final evaluation
    print("Running final evaluation on test set...")
    test_texts = [ex["text"] for ex in test_dataset]
    test_offset_mappings = []
    for ex in test_dataset:
        text = ex["text"]
        encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
        test_offset_mappings.append(encoding["offset_mapping"])
    
    # Temporarily replace compute_metrics for test evaluation
    original_compute_metrics = trainer.compute_metrics
    trainer.compute_metrics = partial(
        compute_metrics_span_level,
        cue_id2label=cue_id2label,
        scope_id2label=scope_id2label,
        tokenizer=tokenizer,
        texts=test_texts,
        offset_mappings=test_offset_mappings,
    )
    
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    
    # Restore original compute_metrics
    trainer.compute_metrics = original_compute_metrics
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"test_eval_results_{timestamp}.txt"
    with open(log_file, "w") as f:
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Test evaluation results saved to {log_file}")
    
    # Save model
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings
    import json
    with open(output_dir / "cue_label2id.json", "w") as f:
        json.dump(cue_label2id, f)
    with open(output_dir / "scope_label2id.json", "w") as f:
        json.dump(scope_label2id, f)


if __name__ == "__main__":
    main()

