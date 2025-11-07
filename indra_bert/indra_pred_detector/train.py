"""Training script for the predicate detector."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .preprocess import (
    RawPredicateExample,
    ID2LABEL,
    LABEL2ID,
    load_and_preprocess_raw_data,
    preprocess_examples_for_model,
)
from .postprocess import spans_from_offsets


class DataCollatorWithDebug(DataCollatorForTokenClassification):
    """Data collator that prints a few token/label examples for inspection."""

    def __init__(self, tokenizer, id2label, max_examples_to_print: int = 3, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.max_examples_to_print = max_examples_to_print
        self._printed = 0

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features)

        if self._printed >= self.max_examples_to_print:
            return batch

        for idx in range(min(len(features), self.max_examples_to_print - self._printed)):
            input_ids = batch["input_ids"][idx]
            labels = batch["labels"][idx]
            attention_mask = batch["attention_mask"][idx]
            token_type_ids = batch.get("token_type_ids")

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
            label_names = [
                self.id2label.get(lbl.item(), "IGN") if lbl.item() != -100 else "PAD"
                for lbl in labels
            ]

            print("\n--- Predicate Detector Training Example ---")
            print("token\t\tlabel\tattn\ttype")
            for j, token in enumerate(tokens):
                attn = attention_mask[j].item()
                label = label_names[j]
                ttype = token_type_ids[idx][j].item() if token_type_ids is not None else "-"
                print(f"{token:15}\t{label:8}\t{attn}\t{ttype}")
            print("-" * 50)

            self._printed += 1
            if self._printed >= self.max_examples_to_print:
                break

        return batch


class TokenClassificationDataset(Dataset):
    """Simple torch Dataset wrapping tokenised inputs."""

    def __init__(self, encoding_dict: dict[str, list]):
        self.encoding = encoding_dict
        self.keys = [
            key
            for key in encoding_dict.keys()
            if key in {"input_ids", "attention_mask", "token_type_ids", "labels"}
        ]

    def __len__(self) -> int:
        return len(self.encoding["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(self.encoding[key][idx])
            for key in self.keys
            if key != "labels"
        }
        labels = torch.tensor(self.encoding["labels"][idx], dtype=torch.long)
        if "attention_mask" in item:
            mask = item["attention_mask"].bool()
            labels = labels.clone()
            labels[~mask] = -100
        item["labels"] = labels
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the predicate detector")
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to the filtered JSONL training file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the trained model and artefacts are stored",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Base encoder model to fine-tune",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data used for validation",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--debug-examples",
        type=int,
        default=3,
        help="Number of tokenised examples to print for debugging",
    )
    return parser.parse_args()


def split_train_val(
    examples: List[RawPredicateExample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[RawPredicateExample], List[RawPredicateExample]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    val_size = int(len(examples) * val_ratio)
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]
    return train_examples, val_examples


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics_span_level(pred_output, eval_dataset: TokenClassificationDataset):
    if "offset_mapping" not in eval_dataset.encoding or "text" not in eval_dataset.encoding:
        return {}

    predictions = pred_output.predictions.argmax(axis=-1)
    label_ids = pred_output.label_ids

    offsets_list = eval_dataset.encoding["offset_mapping"]
    texts = eval_dataset.encoding["text"]

    tp = fp = fn = 0

    for idx, (pred_seq, gold_seq) in enumerate(zip(predictions, label_ids)):
        mask = gold_seq != -100
        pred_seq = pred_seq[mask]
        gold_seq = gold_seq[mask]

        offsets = offsets_list[idx]
        offsets = [offsets[j] for j, flag in enumerate(mask) if flag]
        text = texts[idx]

        pred_spans = set(spans_from_offsets(offsets, pred_seq.tolist(), text))
        gold_spans = set(spans_from_offsets(offsets, gold_seq.tolist(), text))

        tp += len(pred_spans & gold_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_examples = load_and_preprocess_raw_data(args.train_data)
    train_examples, val_examples = split_train_val(raw_examples, args.val_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<e>", "</e>"]})

    train_encoding = preprocess_examples_for_model(
        train_examples,
        tokenizer,
        max_length=args.max_seq_length,
        padding=False,
        truncation=True,
    )
    val_encoding = None
    if val_examples:
        val_encoding = preprocess_examples_for_model(
            val_examples,
            tokenizer,
            max_length=args.max_seq_length,
            padding=False,
            truncation=True,
        )

    train_dataset = TokenClassificationDataset(train_encoding)
    eval_dataset = TokenClassificationDataset(val_encoding) if val_encoding else None

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        logging_steps=50,
        seed=args.seed,
        report_to=[],
    )

    data_collator = DataCollatorWithDebug(
        tokenizer=tokenizer,
        id2label=ID2LABEL,
        max_examples_to_print=args.debug_examples,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(
            lambda output: compute_metrics_span_level(output, eval_dataset)
            if eval_dataset is not None
            else None
        ),
    )

    trainer.train()

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, default=str)


if __name__ == "__main__":
    main()
