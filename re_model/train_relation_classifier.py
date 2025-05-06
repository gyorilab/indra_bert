import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support

from relation_model import RelationClassifier
from prepare_relation_classifier_dataset import load_and_preprocess_raw_data, preprocess_examples_for_model, relationship_types

label2id = {label: i for i, label in enumerate(relationship_types)}
id2label = {v: k for k, v in label2id.items()}

# ---- Metrics ----
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main(args):
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ---- Load and preprocess raw data ----
    examples = load_and_preprocess_raw_data(dataset_path, tokenizer)
    dataset = Dataset.from_list(examples)

    # ---- Split dataset ----
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
    train_dataset = split_dataset["train"]
    temp_dataset = split_dataset["test"]
    val_test_split = temp_dataset.train_test_split(test_size=1/3, seed=42)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    # ---- Tokenize datasets ----
    train_dataset = train_dataset.map(lambda x: preprocess_examples_for_model(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: preprocess_examples_for_model(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: preprocess_examples_for_model(x, tokenizer), batched=True)

    # ---- Model ----
    model = RelationClassifier(args.model_name, num_labels=len(relationship_types))

    # ---- Training ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
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
    parser = argparse.ArgumentParser(description="Train Relation Classifier")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSONL file")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and logs")

    args = parser.parse_args()
    main(args)
