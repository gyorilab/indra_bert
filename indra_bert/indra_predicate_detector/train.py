from datetime import datetime
import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from typing import Any, Dict, List
import torch
from functools import partial

from .preprocess import (
    load_tsv_dataset,
    build_label_mappings,
    preprocess_examples,
)
from .postprocess import extract_trigger_spans_from_encoding


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
                label_names = [
                    self.id2label.get(l.item(), "IGN") if l.item() != -100 else "PAD"
                    for l in labels
                ]

                print("\n--- DEBUG: Training Example ---")
                print(f"{'Token':20} {'Label':20} {'AttnMask':9}")
                print("-" * 60)
                for j, tok in enumerate(tokens):
                    attn = attention_mask[j].item()
                    label = label_names[j]
                    print(f"{tok:20} {label:20} {attn:<9}")
                print("-" * 60)

            self.counter += len(features)

        return batch


def compute_metrics_span_level(eval_preds, id2label, examples):
    """
    Compute span-level metrics for trigger detection.
    Evaluates (trigger_span, role) tuples as the unit of prediction.
    """
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids

    TP, FP, FN = 0, 0, 0

    for i in range(len(predictions)):
        pred_ids = np.argmax(predictions[i], axis=1).tolist()
        gold_ids = labels[i].tolist()

        example = examples[i]
        tokens = example["tokens"]
        offsets = example["offset_mapping"]
        text = example["text"]

        # Extract predicted spans with roles
        pred_spans = extract_trigger_spans_from_encoding(
            tokens, offsets, pred_ids, id2label, text
        )
        # Create set of (start, end, role) tuples for matching
        pred_tuples = {
            (s["start"], s["end"], s.get("role"))
            for s in pred_spans
        }

        # Extract gold spans with roles
        gold_spans = extract_trigger_spans_from_encoding(
            tokens, offsets, gold_ids, id2label, text
        )
        gold_tuples = {
            (s["start"], s["end"], s.get("role"))
            for s in gold_spans
        }

        TP += len(pred_tuples & gold_tuples)
        FP += len(pred_tuples - gold_tuples)
        FN += len(gold_tuples - pred_tuples)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}


def parse_args():
    parser = argparse.ArgumentParser(description="Train predicate detector model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to TSV dataset file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model name or path",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_cached_dataset", action="store_true")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep")
    parser.add_argument("--version", type=str, default="1.0", help="Version of the training script")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dataset_path = output_dir / "cached_dataset"
    cache_label2id_path = cache_dataset_path / "label2id.json"

    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<e>", "</e>"]})

    if args.use_cached_dataset and cache_dataset_path.exists() and cache_label2id_path.exists():
        print("Loading from cache...")
        import json
        cached = DatasetDict.load_from_disk(str(cache_dataset_path))
        train_dataset = cached["train"]
        val_dataset = cached["validation"]
        test_dataset = cached["test"]
        with open(cache_label2id_path, "r") as f:
            label2id = json.load(f)
        id2label = {int(k): v for k, v in json.load(open(cache_dataset_path / "id2label.json")).items()}
    else:
        print("No cache found, processing dataset...")
        # Load raw data
        print(f"Loading dataset from {dataset_path}...")
        raw_examples = load_tsv_dataset(dataset_path)
        print(f"Total examples: {len(raw_examples)}")

        # Build label mappings
        print("Building label mappings...")
        label2id, id2label = build_label_mappings(raw_examples)
        print(f"Labels: {list(label2id.keys())}")

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(raw_examples)

        # Split dataset: 70% train, 20% val, 10% test
        print("Splitting dataset...")
        split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
        train_dataset = split_dataset["train"]
        temp_dataset = split_dataset["test"]
        val_test_split = temp_dataset.train_test_split(test_size=1/3, seed=42)
        val_dataset = val_test_split["train"]
        test_dataset = val_test_split["test"]

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Preprocess datasets
        print("Preprocessing datasets...")
        preprocess_fn = partial(preprocess_examples, tokenizer=tokenizer, label2id=label2id)

        train_dataset = train_dataset.map(
            lambda x: preprocess_fn(x),
            batched=False
        )
        val_dataset = val_dataset.map(
            lambda x: preprocess_fn(x),
            batched=False
        )
        test_dataset = test_dataset.map(
            lambda x: preprocess_fn(x),
            batched=False
        )

        # Save cache
        if args.use_cached_dataset:
            print("Caching preprocessed datasets...")
            cache_dataset_path.mkdir(parents=True, exist_ok=True)
            import json
            cached = DatasetDict(train=train_dataset, validation=val_dataset, test=test_dataset)
            cached.save_to_disk(str(cache_dataset_path))
            with open(cache_label2id_path, "w") as f:
                json.dump(label2id, f, indent=2)
            with open(cache_dataset_path / "id2label.json", "w") as f:
                json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)

    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

    # Load model
    training_config = vars(args).copy()
    training_config['time_created'] = datetime.now().strftime("%Y-%m-%d")
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = len(label2id)
    config.id2label = id2label
    config.label2id = label2id
    config.training_config = training_config

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, config=config
    )
    # Resize token embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Data collator
    data_collator = DataCollatorWithDebug(
        tokenizer=tokenizer,
        id2label=id2label,
        padding=True,
    )

    # Prepare eval examples for span-level metrics
    # Extract examples from validation dataset once
    # Use annotated_text (with <e> tags) since offsets are relative to that
    eval_examples_for_metrics = []
    for i in range(len(dataset_dict["validation"])):
        example = dataset_dict["validation"][i]
        eval_examples_for_metrics.append({
            "tokens": example["tokens"],
            "offset_mapping": example["offset_mapping"],
            "text": example["annotated_text"],  # Use annotated_text since offsets are relative to it
        })
    
    # Metrics function - only span-level metrics based on (trigger_span, role) tuples
    def compute_metrics(p):
        # Compute span-level metrics using (span, role) tuples
        span_metrics = compute_metrics_span_level(p, id2label, eval_examples_for_metrics)
        
        return span_metrics

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()

