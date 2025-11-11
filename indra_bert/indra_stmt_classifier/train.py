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

from .model import MultiHeadStmtClassifier, IGNORE_INDEX
from .preprocess import (
    BINARY_LABEL_MAPPING,
    load_relation_binary_dataset,
    build_relation_subtype_mapping,
    create_dataset_splits,
    tokenize_relation_dataset,
    load_indra_benchmark_dataset,
    build_indra_label_mapping,
    tokenize_indra_dataset,
)


class MultiHeadDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Extract labels before padding
        gate1 = [f.pop("gate1_labels") for f in features]
        gate2 = [f.pop("gate2_labels") for f in features]
        gate3 = [f.pop("gate3_labels") for f in features]

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["gate1_labels"] = torch.tensor(gate1, dtype=torch.long)
        batch["gate2_labels"] = torch.tensor(gate2, dtype=torch.long)
        batch["gate3_labels"] = torch.tensor(gate3, dtype=torch.long)
        return batch


def build_compute_metrics(gate2_size: int, gate3_size: int):
    """
    Metrics over:
      - gate1: has_relation / no_relation
      - gate2: subtype (masked where IGNORE_INDEX)
      - gate3: INDRA label (masked)
    """
    def _metrics(eval_pred):
        predictions, labels = eval_pred

        # predictions is logits from SequenceClassifierOutput: [N, 2 + gate2 + gate3]
        if isinstance(predictions, tuple):
            # Just in case; but with our model it should be a single array.
            predictions = predictions[0]

        combined_logits = np.asarray(predictions)
        gate1_logits = combined_logits[:, :2]
        gate2_logits = combined_logits[:, 2 : 2 + gate2_size]
        gate3_logits = combined_logits[:, 2 + gate2_size : 2 + gate2_size + gate3_size]

        # labels comes as tuple because of label_names
        if isinstance(labels, tuple):
            gate1_labels, gate2_labels, gate3_labels = labels
        else:
            # fallback: treat as gate1-only
            gate1_labels = labels
            gate2_labels = None
            gate3_labels = None

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
    parser.add_argument("--gate2_loss_weight", type=float, default=0.5)
    parser.add_argument("--gate3_loss_weight", type=float, default=0.25)
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

        # ----- Align columns / add missing gates -----
        keep_cols = {
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "gate1_labels",
            "gate2_labels",
            "gate3_labels",
        }

        relation_train = align_columns(relation_train, keep_cols)
        relation_val = align_columns(relation_val, keep_cols)
        relation_test = align_columns(relation_test, keep_cols)

        def ensure_columns(ds: Dataset):
            for col in keep_cols:
                if col not in ds.column_names:
                    filler = np.zeros(len(ds), dtype=np.int64)
                    if col.startswith("gate"):
                        filler.fill(IGNORE_INDEX)
                    ds = ds.add_column(col, filler)
            return align_columns(ds, keep_cols)

        indra_train = ensure_columns(indra_train)
        indra_val = ensure_columns(indra_val)
        indra_test = ensure_columns(indra_test)

        # ----- Merge for multitask training -----
        train_dataset = concatenate_datasets([relation_train, indra_train]).shuffle(args.seed)
        eval_dataset = concatenate_datasets([relation_val, indra_val]).shuffle(args.seed)
        test_dataset = concatenate_datasets([relation_test, indra_test]).shuffle(args.seed)

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            DatasetDict(train=train_dataset, eval=eval_dataset, test=test_dataset).save_to_disk(str(cache_dir))
            with open(cache_dir / "relation_subtype2id.json", "w", encoding="utf-8") as fh:
                json.dump(subtype2id, fh, indent=2)
            with open(cache_dir / "indra_label2id.json", "w", encoding="utf-8") as fh:
                json.dump(indra2id, fh, indent=2)

    # ----- Model -----
    config = AutoConfig.from_pretrained(args.model_name)
    model = MultiHeadStmtClassifier.from_pretrained(
        args.model_name,
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
        label_names=["gate1_labels", "gate2_labels", "gate3_labels"],
    )

    data_collator = MultiHeadDataCollator(tokenizer)
    compute_metrics = build_compute_metrics(len(subtype2id), len(indra2id))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_strategy != "no" else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_strategy != "no" else None,
    )

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


if __name__ == "__main__":
    main()
