import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support

from indra_stmt_classifier_model.relation_model import BertForIndraStmtClassification
from indra_stmt_classifier_model.preprocess import (
    load_and_preprocess_raw_data,
    preprocess_examples_for_model,
)
from indra_stmt_classifier_model.weighted_trainer import WeightedTrainer, compute_class_weights


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def main(args):
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ---- Load and preprocess raw data ----
    examples, stmt2id = load_and_preprocess_raw_data(dataset_path)
    id2stmt = {v: k for k, v in stmt2id.items()}

    dataset = Dataset.from_list(examples)

    # ---- Split dataset ----
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42, shuffle=True)
    train_dataset = split_dataset["train"]
    temp_dataset = split_dataset["test"]
    val_test_split = temp_dataset.train_test_split(test_size=1/3, seed=42, shuffle=True)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    # ---- Tokenize datasets ----
    train_dataset = train_dataset.map(
        lambda x: preprocess_examples_for_model(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_examples_for_model(x, tokenizer), batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_examples_for_model(x, tokenizer), batched=True
    )

    # ---- Model ----
    model = BertForIndraStmtClassification.from_pretrained_with_labels(
        pretrained_model_name=args.model_name,
        label2id=stmt2id,
        id2label=id2stmt,
    )

    # ---- Training ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
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
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    # ---- Final evaluation ----
    print("Running final evaluation on test set...")
    test_metrics = trainer.evaluate(test_dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"test_eval_results_{timestamp}.txt"
    with open(log_file, "w") as f:
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Test evaluation results saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train INDRA Statement Classifier")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSONL file")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and logs")
    args = parser.parse_args()
    main(args)
