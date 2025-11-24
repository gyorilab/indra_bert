import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import PredictionOutput

from .model import MultiHeadStmtClassifier, IGNORE_INDEX
from .preprocess import (
    BINARY_LABEL_MAPPING,
    IGNORE_INDEX as PREPROCESS_IGNORE_INDEX,
    load_relation_binary_dataset,
    build_relation_subtype_mapping,
    create_dataset_splits,
    tokenize_relation_dataset,
    load_indra_benchmark_dataset,
    build_indra_label_mapping,
    tokenize_indra_dataset,
    load_hrt_dataset,
    build_gate4_label_mapping,
    tokenize_hrt_dataset,
)


class MultiHeadTrainer(Trainer):
    """Custom Trainer that includes gate4_logits in predictions for metrics computation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate4_predictions_list = []
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override to include gate4_logits in predictions.
        """
        has_labels = any(k in inputs for k in ["gate1_labels", "gate2_labels", "gate3_labels", "gate4_labels"])
        
        # Call model forward to get gate4_logits
        if not prediction_loss_only:
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss if has_labels else None
                
                # Get predictions (main logits for gate1-3) - keep as tensor for evaluation loop
                predictions = outputs.logits.detach()
                
                # Get gate4_logits if available and store separately (as numpy for metrics)
                if hasattr(outputs, 'gate4_logits'):
                    gate4_predictions = outputs.gate4_logits.detach().cpu().numpy()  
                    if not hasattr(self, '_current_gate4_preds'):
                        self._current_gate4_preds = []
                    self._current_gate4_preds.append(gate4_predictions)
                
                # Get labels using parent's logic (returns proper format for evaluation loop)
                # We need to extract labels in the same way the parent does
                labels = None
                if has_labels:
                    # Extract labels as tensors (not numpy) to match parent's format
                    labels = tuple(
                        inputs.get(name).detach() if name in inputs else None
                        for name in ["gate1_labels", "gate2_labels", "gate3_labels", "gate4_labels"]
                    )
                    # Filter out None values and convert to single tensor if only one
                    labels = tuple(l for l in labels if l is not None)
                    if len(labels) == 1:
                        labels = labels[0]
                    elif len(labels) == 0:
                        labels = None
                
                return (loss, predictions, labels)
        else:
            # For loss-only, use parent implementation
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override to make gate4_predictions available to compute_metrics."""
        # Clear previous predictions
        self._current_gate4_preds = []
        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        # Clear after evaluation
        if hasattr(self, '_current_gate4_preds'):
            delattr(self, '_current_gate4_preds')
        return result


class MultiHeadDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Extract labels before padding
        gate1 = [f.pop("gate1_labels") for f in features]
        gate2 = [f.pop("gate2_labels") for f in features]
        gate3 = [f.pop("gate3_labels") for f in features]
        gate4 = [f.pop("gate4_labels", None) for f in features]  # token-level labels (may be None)

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["gate1_labels"] = torch.tensor(gate1, dtype=torch.long)
        batch["gate2_labels"] = torch.tensor(gate2, dtype=torch.long)
        batch["gate3_labels"] = torch.tensor(gate3, dtype=torch.long)
        
        # Pad gate4 labels (token-level) to match sequence length
        max_length = batch["input_ids"].shape[1]
        gate4_padded = []
        for labels in gate4:
            if labels is None:
                # If gate4_labels is missing, create a list of IGNORE_INDEX
                padded = [PREPROCESS_IGNORE_INDEX] * max_length
            elif len(labels) < max_length:
                # Pad with IGNORE_INDEX
                padded = labels + [PREPROCESS_IGNORE_INDEX] * (max_length - len(labels))
            else:
                padded = labels[:max_length]
            gate4_padded.append(padded)
        batch["gate4_labels"] = torch.tensor(gate4_padded, dtype=torch.long)
        
        return batch


def extract_spans_from_token_labels(token_labels, id2label):
    """
    Extract spans from token-level BIO labels.
    Returns list of (start_token_idx, end_token_idx) tuples.
    """
    spans = []
    current_span = None
    
    for i, label_id in enumerate(token_labels):
        label = id2label.get(label_id, "O")
        
        if label == "B-trigger":
            # Start of new span
            if current_span is not None:
                spans.append(current_span)
            current_span = (i, i + 1)
        elif label == "I-trigger":
            # Continue current span
            if current_span is not None:
                current_span = (current_span[0], i + 1)
        else:
            # O - end current span
            if current_span is not None:
                spans.append(current_span)
                current_span = None
    
    # Don't forget the last span
    if current_span is not None:
        spans.append(current_span)
    
    return spans


def build_compute_metrics(gate2_size: int, gate3_size: int, gate4_id2label: dict = None, trainer=None):
    """
    Metrics over:
      - gate1: has_relation / no_relation
      - gate2: subtype (masked where IGNORE_INDEX)
      - gate3: INDRA label (masked)
      - gate4: predicate spans (span-level, not token-level)
    """
    def _metrics(eval_pred):
        predictions, labels = eval_pred

        # predictions is logits from SequenceClassifierOutput: [N, 2 + gate2 + gate3]
        # But we also need gate4_logits which are stored separately
        # Actually, the Trainer will pass the model output, which includes gate4_logits
        # But compute_metrics receives predictions from model outputs...
        # We need to handle this differently - gate4_logits are in a separate output
        
        # For now, handle the sequence classification predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        combined_logits = np.asarray(predictions)
        gate1_logits = combined_logits[:, :2]
        gate2_logits = combined_logits[:, 2 : 2 + gate2_size]
        gate3_logits = combined_logits[:, 2 + gate2_size : 2 + gate2_size + gate3_size]

        # labels comes as tuple because of label_names
        if isinstance(labels, tuple):
            gate1_labels, gate2_labels, gate3_labels, gate4_labels = labels
        else:
            # fallback: treat as gate1-only
            gate1_labels = labels
            gate2_labels = None
            gate3_labels = None
            gate4_labels = None

        metrics = {}

        def head_metrics(head_logits, head_labels, prefix: str):
            if head_labels is None:
                return
            head_labels = np.asarray(head_labels)
            mask = head_labels != IGNORE_INDEX
            if not np.any(mask):
                metrics[f"{prefix}_support"] = 0
                return

            y_true = head_labels[mask]
            y_pred = np.argmax(head_logits[mask], axis=-1)

            metrics[f"{prefix}_accuracy"] = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
            metrics[f"{prefix}_precision"] = precision
            metrics[f"{prefix}_recall"] = recall
            metrics[f"{prefix}_f1"] = f1
            metrics[f"{prefix}_support"] = int(mask.sum())

        head_metrics(gate1_logits, gate1_labels, "gate1")
        head_metrics(gate2_logits, gate2_labels, "gate2")
        head_metrics(gate3_logits, gate3_labels, "gate3")
        
        # Gate4: span-level metrics (compare predicted spans vs gold spans)
        if gate4_labels is not None and gate4_id2label is not None:
            gate4_labels = np.asarray(gate4_labels)
            
            # Get gate4_predictions from trainer instance if available
            gate4_logits = None
            if trainer is not None and hasattr(trainer, "_current_gate4_preds"):
                if trainer._current_gate4_preds:
                    gate4_batch_list = trainer._current_gate4_preds

                    # Find global max sequence length across batches
                    max_len = max(arr.shape[1] for arr in gate4_batch_list)

                    padded_batches = []
                    for arr in gate4_batch_list:
                        bsz, seq_len, num_labels = arr.shape
                        if seq_len < max_len:
                            pad = np.zeros(
                                (bsz, max_len - seq_len, num_labels),
                                dtype=arr.dtype,
                            )
                            arr_padded = np.concatenate([arr, pad], axis=1)
                        elif seq_len > max_len:
                            # Just in case some batch is longer, clip it
                            arr_padded = arr[:, :max_len, :]
                        else:
                            arr_padded = arr
                        padded_batches.append(arr_padded)

                    gate4_logits = np.concatenate(padded_batches, axis=0)
            
            if gate4_logits is not None:
                gate4_preds = np.argmax(gate4_logits, axis=-1)  # [B, L]
                
                TP, FP, FN = 0, 0, 0
                
                for i in range(len(gate4_preds)):
                    # Extract predicted spans
                    pred_spans = extract_spans_from_token_labels(gate4_preds[i], gate4_id2label)
                    # Extract gold spans (only where labels are not IGNORE_INDEX)
                    gold_labels = gate4_labels[i]
                    mask = gold_labels != IGNORE_INDEX
                    if mask.any():
                        gold_labels_masked = gold_labels[mask]
                        # Need to adjust indices since we masked
                        # Actually, let's extract spans from full labels and filter later
                        gold_spans = extract_spans_from_token_labels(gate4_labels[i], gate4_id2label)
                        
                        # Convert to sets for comparison
                        pred_span_set = set(pred_spans)
                        gold_span_set = set(gold_spans)
                        
                        TP += len(pred_span_set & gold_span_set)
                        FP += len(pred_span_set - gold_span_set)
                        FN += len(gold_span_set - pred_span_set)
                
                if TP + FP + FN > 0:
                    precision = TP / (TP + FP + 1e-8)
                    recall = TP / (TP + FN + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    
                    metrics["gate4_span_precision"] = precision
                    metrics["gate4_span_recall"] = recall
                    metrics["gate4_span_f1"] = f1
                    metrics["gate4_span_TP"] = TP
                    metrics["gate4_span_FP"] = FP
                    metrics["gate4_span_FN"] = FN

        return metrics

    return _metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relation_path",
        type=Path,
        default=Path("data/train/statement_classification/combined/relation_binary.jsonl"),
    )
    parser.add_argument(
        "--indra_path",
        type=Path,
        default=Path(
            "data/train/indra_benchmark_annotated_data/"
            "indra_benchmark_corpus_annotated_stratified_sample_2000.jsonl"
        ),
    )
    parser.add_argument(
        "--hrt_path",
        type=Path,
        default=None,
        help="Path to HRT dataset TSV file (optional, for gate4 predicate detection)",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--model_name",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gate1_loss_weight", type=float, default=1.0)
    parser.add_argument("--gate2_loss_weight", type=float, default=1.0)
    parser.add_argument("--gate3_loss_weight", type=float, default=1.0)
    parser.add_argument("--gate4_loss_weight", type=float, default=1.0)
    parser.add_argument("--eval_strategy", type=str, default="epoch")  # "no", "steps", or "epoch"
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Optional directory for caching tokenized datasets.",
    )
    parser.add_argument(
        "--use_cached_dataset",
        action="store_true",
        help="Load tokenized datasets from cache_dir if available.",
    )
    return parser.parse_args()


def align_columns(dataset: Dataset, keep_columns):
    drop_cols = [c for c in dataset.column_names if c not in keep_columns]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)
    return dataset


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<e>", "</e>"]})

    cache_dir = args.cache_dir
    if cache_dir is None and args.use_cached_dataset:
        cache_dir = args.output_dir / "cached_dataset"
    if cache_dir is not None:
        cache_dir = cache_dir.resolve()
    args.cache_dir = cache_dir

    if args.use_cached_dataset and cache_dir and cache_dir.exists():
        dataset_dict = DatasetDict.load_from_disk(str(cache_dir))
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["eval"]
        test_dataset = dataset_dict["test"]
        with open(cache_dir / "relation_subtype2id.json", "r", encoding="utf-8") as fh:
            subtype2id = json.load(fh)
        with open(cache_dir / "indra_label2id.json", "r", encoding="utf-8") as fh:
            indra2id = json.load(fh)
        gate4_label2id_path = cache_dir / "gate4_label2id.json"
        if gate4_label2id_path.exists():
            with open(gate4_label2id_path, "r", encoding="utf-8") as fh:
                gate4_label2id = json.load(fh)
        else:
            gate4_label2id, _ = build_gate4_label_mapping()
    else:
        # ----- Relation dataset (Gate1 + Gate2) -----
        relation_examples = load_relation_binary_dataset(args.relation_path)
        subtype2id = build_relation_subtype_mapping(relation_examples)
        relation_splits = create_dataset_splits(relation_examples, seed=args.seed)

        relation_train = tokenize_relation_dataset(relation_splits["train"], tokenizer, subtype2id)
        relation_val = tokenize_relation_dataset(relation_splits["validation"], tokenizer, subtype2id)
        relation_test = tokenize_relation_dataset(relation_splits["test"], tokenizer, subtype2id)

        # ----- INDRA dataset (Gate3) -----
        indra_examples = load_indra_benchmark_dataset(args.indra_path)
        indra2id = build_indra_label_mapping(indra_examples)
        indra_dataset = Dataset.from_list(indra_examples)

        indra_split = indra_dataset.train_test_split(test_size=0.2, seed=args.seed)
        indra_val_test = indra_split["test"].train_test_split(test_size=0.5, seed=args.seed)

        indra_train = tokenize_indra_dataset(indra_split["train"], tokenizer, indra2id)
        indra_val = tokenize_indra_dataset(indra_val_test["train"], tokenizer, indra2id)
        indra_test = tokenize_indra_dataset(indra_val_test["test"], tokenizer, indra2id)

        # ----- HRT dataset (Gate4) -----
        gate4_label2id = None
        gate4_id2label = None
        hrt_train = hrt_val = hrt_test = None
        if args.hrt_path and args.hrt_path.exists():
            hrt_examples = load_hrt_dataset(args.hrt_path)
            gate4_label2id, gate4_id2label = build_gate4_label_mapping()
            hrt_dataset = Dataset.from_list(hrt_examples)
            
            hrt_split = hrt_dataset.train_test_split(test_size=0.2, seed=args.seed)
            hrt_val_test = hrt_split["test"].train_test_split(test_size=0.5, seed=args.seed)
            
            hrt_train = tokenize_hrt_dataset(hrt_split["train"], tokenizer, gate4_label2id)
            hrt_val = tokenize_hrt_dataset(hrt_val_test["train"], tokenizer, gate4_label2id)
            hrt_test = tokenize_hrt_dataset(hrt_val_test["test"], tokenizer, gate4_label2id)
        else:
            # Build empty label mapping if HRT not provided
            gate4_label2id, gate4_id2label = build_gate4_label_mapping()

        # ----- Align columns / add missing gates -----
        keep_cols = {
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "gate1_labels",
            "gate2_labels",
            "gate3_labels",
            "gate4_labels",
        }

        relation_train = align_columns(relation_train, keep_cols)
        relation_val = align_columns(relation_val, keep_cols)
        relation_test = align_columns(relation_test, keep_cols)

        def ensure_columns(ds: Dataset, is_gate4=False):
            for col in keep_cols:
                if col not in ds.column_names:
                    if col == "gate4_labels" and is_gate4:
                        # For gate4, we need token-level labels (list of lists)
                        # This should already be handled in tokenize_hrt_dataset
                        continue
                    # For token_type_ids, we need to create lists matching sequence length
                    if col == "token_type_ids":
                        # Get sequence length from input_ids
                        if "input_ids" in ds.column_names:
                            # Create list of zeros matching each sequence length
                            filler = [[0] * len(ids) for ids in ds["input_ids"]]
                        else:
                            # Fallback: create list of zeros with max_length
                            filler = [[0] * 512 for _ in range(len(ds))]
                    else:
                        filler = np.zeros(len(ds), dtype=np.int64)
                        if col.startswith("gate"):
                            if col == "gate4_labels":
                                # For gate4, create list of IGNORE_INDEX lists
                                filler = [[IGNORE_INDEX] * 512 for _ in range(len(ds))]
                            else:
                                filler.fill(IGNORE_INDEX)
                    ds = ds.add_column(col, filler)
            return align_columns(ds, keep_cols)

        indra_train = ensure_columns(indra_train)
        indra_val = ensure_columns(indra_val)
        indra_test = ensure_columns(indra_test)
        
        if hrt_train is not None:
            hrt_train = ensure_columns(hrt_train, is_gate4=True)
            hrt_val = ensure_columns(hrt_val, is_gate4=True)
            hrt_test = ensure_columns(hrt_test, is_gate4=True)

        # ----- Merge for multitask training -----
        train_datasets = [relation_train, indra_train]
        eval_datasets = [relation_val, indra_val]
        test_datasets = [relation_test, indra_test]
        
        if hrt_train is not None:
            train_datasets.append(hrt_train)
            eval_datasets.append(hrt_val)
            test_datasets.append(hrt_test)
        
        train_dataset = concatenate_datasets(train_datasets).shuffle(args.seed)
        eval_dataset = concatenate_datasets(eval_datasets).shuffle(args.seed)
        test_dataset = concatenate_datasets(test_datasets).shuffle(args.seed)

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            DatasetDict(train=train_dataset, eval=eval_dataset, test=test_dataset).save_to_disk(str(cache_dir))
            with open(cache_dir / "relation_subtype2id.json", "w", encoding="utf-8") as fh:
                json.dump(subtype2id, fh, indent=2)
            with open(cache_dir / "indra_label2id.json", "w", encoding="utf-8") as fh:
                json.dump(indra2id, fh, indent=2)
            with open(cache_dir / "gate4_label2id.json", "w", encoding="utf-8") as fh:
                json.dump(gate4_label2id, fh, indent=2)

    # ----- Model -----
    gate4_num_labels = len(gate4_label2id) if gate4_label2id else 3
    config = AutoConfig.from_pretrained(args.model_name)
    model = MultiHeadStmtClassifier.from_pretrained(
        args.model_name,
        gate4_num_labels=gate4_num_labels,
        gate4_loss_weight=args.gate4_loss_weight,
        config=config,
        gate2_num_labels=len(subtype2id),
        gate3_num_labels=len(indra2id),
        gate1_loss_weight=args.gate1_loss_weight,
        gate2_loss_weight=args.gate2_loss_weight,
        gate3_loss_weight=args.gate3_loss_weight,
    )
    model.resize_token_embeddings(len(tokenizer))

    # ----- Training args -----
    eval_strategy = args.eval_strategy
    save_strategy = "epoch" if eval_strategy != "no" else "no"
    load_best = eval_strategy != "no"

    save_total_limit = 2 if load_best else 1

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy=eval_strategy,              # alias for evaluation_strategy
        save_strategy=save_strategy,
        logging_strategy="epoch" if eval_strategy != "no" else "no",
        report_to=[],
        seed=args.seed,
        load_best_model_at_end=load_best,
        metric_for_best_model="gate1_f1",
        greater_is_better=True,
        label_names=["gate1_labels", "gate2_labels", "gate3_labels", "gate4_labels"],
        save_total_limit=save_total_limit,
    )

    data_collator = MultiHeadDataCollator(tokenizer)
    gate4_id2label = {v: k for k, v in gate4_label2id.items()} if gate4_label2id else None
    
    # Create trainer first, then build metrics with reference to trainer
    trainer = MultiHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_strategy != "no" else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,  # Set after creation
    )
    
    # Build metrics with trainer reference
    compute_metrics = build_compute_metrics(len(subtype2id), len(indra2id), gate4_id2label, trainer)
    trainer.compute_metrics = compute_metrics if eval_strategy != "no" else None

    # ----- Train -----
    trainer.train()

    # ----- Eval & Test -----
    if eval_strategy != "no":
        metrics = trainer.evaluate(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        test_metrics = trainer.evaluate(test_dataset)
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    # ----- Save model & metadata -----
    trainer.save_state()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(args.output_dir / "relation_subtype2id.json", "w", encoding="utf-8") as fh:
        json.dump(subtype2id, fh, indent=2)
    with open(args.output_dir / "indra_label2id.json", "w", encoding="utf-8") as fh:
        json.dump(indra2id, fh, indent=2)
    with open(args.output_dir / "binary_label2id.json", "w", encoding="utf-8") as fh:
        json.dump(BINARY_LABEL_MAPPING, fh, indent=2)
    with open(args.output_dir / "gate4_label2id.json", "w", encoding="utf-8") as fh:
        json.dump(gate4_label2id, fh, indent=2)


if __name__ == "__main__":
    main()
