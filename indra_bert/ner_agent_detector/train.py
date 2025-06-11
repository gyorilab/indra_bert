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
    load_and_preprocess_from_raw_data,
    build_label_mappings,
    preprocess_examples
)

from transformers import DataCollatorForTokenClassification
from typing import Any, Dict, List
import torch

from .postprocess import extract_spans_from_encoding

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
                token_type_ids = batch["token_type_ids"]

                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                label_names = [self.id2label.get(l.item(), "IGN") if l.item() != -100 else "PAD" for l in labels]

                print("\n--- DEBUG: Training Example ---")
                print("-" * 50)
                for j, tok in enumerate(tokens):
                    attn = attention_mask[j].item()
                    label = label_names[j]
                    ttype = token_type_ids[i][j].item() if token_type_ids is not None else "-"
                    print(f"{tok:15} {label:12} {attn:<9} {ttype}")
                print("-" * 50)

            self.counter += len(features)

        return batch
    
# ---- Metrics ----
def compute_metrics_span_level(eval_preds: EvalPrediction, inputs, id2label, texts):
    """
    Compute span-level metrics using predictions from the Trainer.
    """
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids

    TP, FP, FN = 0, 0, 0

    for i in range(len(predictions)):
        pred_ids = np.argmax(predictions[i], axis=1).tolist()
        gold_ids = labels[i].tolist()

        tokens = inputs[i]["tokens"]
        offsets = inputs[i]["offset_mapping"]
        text = texts[i]

        pred_spans = {
            (s["start"], s["end"], s["text"])
            for s in extract_spans_from_encoding(tokens, offsets, pred_ids, id2label, text)
        }

        gold_spans = {
            (s["start"], s["end"], s["text"])
            for s in extract_spans_from_encoding(tokens, offsets, gold_ids, id2label, text)
        }

        TP += len(pred_spans & gold_spans)
        FP += len(pred_spans - gold_spans)
        FN += len(gold_spans - pred_spans)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}

def parse_args():
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--version", type=str, default="1.0", help="Version of the training script")
    parser.add_argument("--use_cached_dataset", action="store_true", help="Use cached dataset if available")
    args = parser.parse_args()
    return args

# ---- Main function ----
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dataset_path = output_dir / "cached_dataset"
    cache_label2id_path = cache_dataset_path / "label2id.npy"

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.use_cached_dataset and cache_dataset_path.exists() and cache_label2id_path.exists():
        print("Loading from cache...")
        cached = load_from_disk(cache_dataset_path)
        train_dataset = cached["train"]
        val_dataset = cached["validation"]
        test_dataset = cached["test"]
        with open(cache_label2id_path, "rb") as f:
            label2id = np.load(f, allow_pickle=True).item()
        id2label = {v: k for k, v in label2id.items()}
    else:
        print("No cache found, processing dataset...")
        # ---- Load raw data ----
        train_raw = load_and_preprocess_from_raw_data(Path(args.train_data))
        eval_raw = load_and_preprocess_from_raw_data(Path(args.val_data))
        test_raw = load_and_preprocess_from_raw_data(Path(args.test_data))

        # ---- Build label mappings from dataset ----
        label2id, id2label = build_label_mappings(train_raw + eval_raw + test_raw)

        # ---- Convert to HuggingFace Dataset ----
        train_dataset = Dataset.from_list(train_raw)
        val_dataset = Dataset.from_list(eval_raw)
        test_dataset = Dataset.from_list(test_raw)

        # ---- Preprocess for model ----
        train_dataset = train_dataset.map(
            lambda x: preprocess_examples(x, tokenizer, label2id),
            batched=False
        )
        val_dataset = val_dataset.map(
            lambda x: preprocess_examples(x, tokenizer, label2id),
            batched=False
        )
        test_dataset = test_dataset.map(
            lambda x: preprocess_examples(x, tokenizer, label2id),
            batched=False
        )
        # ---- Save cache ----
        # Save to cache
        cached = DatasetDict(train=train_dataset, validation=val_dataset, test=test_dataset)
        cached.save_to_disk(cache_dataset_path)
        with open(cache_label2id_path, "wb") as f:
            np.save(f, label2id)

    # ---- Load model ----
    training_config = vars(args).copy()
    training_config['time_created'] = datetime.now().strftime("%Y-%m-%d")
    config = AutoConfig.from_pretrained(
            args.model_name,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label
    )
    config.training_config = training_config

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        config=config
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        print("Resizing token embeddings to match tokenizer size...")
        print("Old embedding size:", model.get_input_embeddings().num_embeddings)
        print("New embedding size:", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir='./logs',
    )


    data_collator = DataCollatorWithDebug(
        tokenizer=tokenizer,
        id2label=id2label,
        max_examples_to_print=3,  # Print only first 3 batches
        padding="longest",
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(
            compute_metrics_span_level,
            inputs=val_dataset,  # <- pass preprocessed dataset with offsets, tokens, text
            id2label=id2label,
            texts=[ex["text"] for ex in val_dataset]
        )
    )

    trainer.train()

    # ---- Final evaluation ----
    print("Running final evaluation on the test set...")
    test_metrics = trainer.evaluate(test_dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"test_eval_results_{timestamp}.txt"
    with open(log_file, "w") as f:
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Test evaluation results saved to {log_file}")


if __name__ == "__main__":
    main()
