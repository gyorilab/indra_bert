from datetime import datetime
import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification, 
                          AutoConfig, Trainer, TrainingArguments)
from sklearn.metrics import precision_recall_fscore_support


from .preprocess import (
    load_and_preprocess_from_raw_data,
    build_label_mappings,
    preprocess_examples_from_dataset,
    SpecialTokenOffsetFixTokenizer,
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
                token_type_ids = batch.get("token_type_ids", None)

                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                label_names = [self.id2label.get(l.item(), "IGN") if l.item() != -100 else "PAD" for l in labels]

                print("\n--- DEBUG: Training Example ---")
                print(f"{'Token':15} {'Label':12} {'AttnMask':9} {'TokenType':9}")
                print("-" * 50)
                for j, tok in enumerate(tokens):
                    attn = attention_mask[j].item()
                    label = label_names[j]
                    ttype = token_type_ids[j].item() if token_type_ids is not None else "-"
                    print(f"{tok:15} {label:12} {attn:<9} {ttype}")
                print("-" * 50)

            self.counter += len(features)

        return batch
    
# ---- Metrics ----
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions, true_labels = [], []
    for pred, label in zip(predictions, labels):
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                true_predictions.append(p_)
                true_labels.append(l_)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average="macro", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}

def parse_args():
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--use_cached_dataset", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dataset_path = output_dir / "cached_dataset"
    cache_label2id_path = cache_dataset_path / "label2id.npy"

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
    special_tokenizer = SpecialTokenOffsetFixTokenizer(tokenizer)

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
        raw_examples = load_and_preprocess_from_raw_data(dataset_path)

        # ---- Build label mappings from dataset ----
        label2id, _ = build_label_mappings(raw_examples)
        id2label = {v: k for k, v in label2id.items()}

        # ---- Convert to HuggingFace Dataset ----
        dataset = Dataset.from_list(raw_examples)

        # ---- Split dataset ----
        split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
        train_dataset = split_dataset["train"]
        temp_dataset = split_dataset["test"]
        val_test_split = temp_dataset.train_test_split(test_size=1/3, seed=42)
        val_dataset = val_test_split["train"]
        test_dataset = val_test_split["test"]

        # ---- Preprocess for model ----
        train_dataset = train_dataset.map(
            lambda x: preprocess_examples_from_dataset(x, special_tokenizer, label2id),
            batched=False
        )
        val_dataset = val_dataset.map(
            lambda x: preprocess_examples_from_dataset(x, special_tokenizer, label2id),
            batched=False
        )
        test_dataset = test_dataset.map(
            lambda x: preprocess_examples_from_dataset(x, special_tokenizer, label2id),
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
    if model.get_input_embeddings().num_embeddings != len(special_tokenizer.tokenizer):
        print("Resizing token embeddings to match tokenizer size...")
        print("Old embedding size:", model.get_input_embeddings().num_embeddings)
        print("New embedding size:",len(special_tokenizer.tokenizer))
        model.resize_token_embeddings(len(special_tokenizer.tokenizer))

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
        tokenizer=special_tokenizer.tokenizer,
        id2label=id2label,
        max_examples_to_print=3,  # Print only first 3 batches
        padding=True,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=special_tokenizer.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
