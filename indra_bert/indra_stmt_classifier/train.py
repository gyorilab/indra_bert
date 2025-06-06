import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import Counter

from datasets import Dataset, concatenate_datasets, DatasetDict, load_from_disk
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support

from .bert_classification_head import EntitySemanticsUnawareHead
from .preprocess import (
    load_and_preprocess_raw_data,
    preprocess_examples_for_model,
    preprocess_negative_examples_for_model
)
from .weighted_trainer import WeightedTrainer, compute_class_weights


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.default_collator = DataCollatorWithPadding(tokenizer)
        self.print_counter = 0

    def __call__(self, features):
        batch = self.default_collator(features)
        if self.print_counter < 3:
            print("Example batch item:", self.tokenizer.decode(batch['input_ids'][0]))
            self.print_counter += 1
        return batch


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--use_cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dataset_path = output_dir / "cached_dataset"
    cache_stmt_path = output_dir / "stmt2id.npy"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<e>', '</e>']})

    if args.use_cache and cache_dataset_path.exists() and cache_stmt_path.exists():
        print("Loading from cache...")
        cached = load_from_disk(cache_dataset_path)
        train_dataset = cached["train"]
        val_dataset = cached["validation"]
        test_dataset = cached["test"]
        with open(cache_stmt_path, "rb") as f:
            stmt2id = np.load(f, allow_pickle=True).item()
        id2stmt = {v: k for k, v in stmt2id.items()}
    else:
        print("No cache found, processing dataset...")
        examples, stmt2id = load_and_preprocess_raw_data(dataset_path)
        id2stmt = {v: k for k, v in stmt2id.items()}
        dataset = Dataset.from_list(examples)

        split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
        train_dataset = split_dataset["train"]
        temp_dataset = split_dataset["test"]
        val_test_split = temp_dataset.train_test_split(test_size=1 / 3, seed=42)
        val_dataset = val_test_split["train"]
        test_dataset = val_test_split["test"]

        # Tokenize positives
        train_pos = train_dataset.map(lambda x: preprocess_examples_for_model(x, tokenizer), batched=True)
        val_pos = val_dataset.map(lambda x: preprocess_examples_for_model(x, tokenizer), batched=True)
        test_pos = test_dataset.map(lambda x: preprocess_examples_for_model(x, tokenizer), batched=True)

        # Generate and tokenize negatives
        train_neg = train_dataset.map(
            preprocess_negative_examples_for_model,
            batched=True,
            remove_columns=train_dataset.column_names,
            fn_kwargs={"stmt2id": stmt2id, "tokenizer": tokenizer}
        )
        val_neg = val_dataset.map(
            preprocess_negative_examples_for_model,
            batched=True,
            remove_columns=val_dataset.column_names,
            fn_kwargs={"stmt2id": stmt2id, "tokenizer": tokenizer}
        )
        test_neg = test_dataset.map(
            preprocess_negative_examples_for_model,
            batched=True,
            remove_columns=test_dataset.column_names,
            fn_kwargs={"stmt2id": stmt2id, "tokenizer": tokenizer}
        )

        # Sample negatives (k = 1)
        k = 4
        train_neg = train_neg.shuffle(seed=42).select(range(min(len(train_neg), k * len(train_pos))))
        val_neg = val_neg.shuffle(seed=42).select(range(min(len(val_neg), k * len(val_pos))))
        test_neg = test_neg.shuffle(seed=42).select(range(min(len(test_neg), k * len(test_pos))))

        # Combine
        train_dataset = concatenate_datasets([train_pos, train_neg])
        val_dataset = concatenate_datasets([val_pos, val_neg])
        test_dataset = concatenate_datasets([test_pos, test_neg])

        # Save to cache
        cached = DatasetDict(train=train_dataset, validation=val_dataset, test=test_dataset)
        cached.save_to_disk(cache_dataset_path)
        with open(cache_stmt_path, "wb") as f:
            np.save(f, stmt2id)

    # ---- Log dataset sizes ----
    label_counts = Counter(train_dataset["labels"])
    print("Label distribution in training data:", label_counts) 

    # ---- Model Setup ----
    model = EntitySemanticsUnawareHead.from_pretrained_with_labels(
        pretrained_model_name=args.model_name,
        label2id=stmt2id,
        id2label=id2stmt,
    )
    model.resize_token_embeddings(len(tokenizer))

    # ---- Training Setup ----
    training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            num_train_epochs=8,
            weight_decay=0.01,
            save_total_limit=1,
            logging_dir="./logs",
    )

    class_weights = compute_class_weights(train_dataset)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollator(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # ---- Train and Evaluate ----
    trainer.train()

    # ---- Final evaluation ----
    print("Running final evaluation on test set...")
    test_metrics = trainer.evaluate(test_dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"test_eval_results_{timestamp}.txt"
    with open(log_file, "w") as f:
        f.write(f"Name: {args.model_name}\n")
        f.write(f"Label distribution in training data: {label_counts}\n")
        f.write("\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Test evaluation results saved to {log_file}")


if __name__ == "__main__":
    main()
