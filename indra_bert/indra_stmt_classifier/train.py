import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support

from .bert_classification_head import BertForIndraStmtClassification
from .preprocess import (
    load_and_preprocess_raw_data,
    preprocess_examples_for_model,
    preprocess_negative_examples_for_model
)
from .weighted_trainer import WeightedTrainer, compute_class_weights

from transformers import DataCollatorWithPadding

class DataCollatorWithEntitySpans:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.default_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        # Separate entity token spans from the rest
        entity_token_spans = [f.pop("entity_token_spans") for f in features]

        # Let the default collator handle everything else
        batch = self.default_collator(features)

        # Now manually add back variable-length entity_token_spans (still list[list[list[int]]])
        batch["entity_token_spans"] = entity_token_spans

        return batch


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

    # Add entity tags as special tokens
    special_tokens_dict = {'additional_special_tokens': ['<e>', '</e>']}
    tokenizer.add_special_tokens(special_tokens_dict)

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
    train_dataset_positive = train_dataset.map(
        lambda x: preprocess_examples_for_model(x, tokenizer), batched=True
    )
    val_dataset_positive = val_dataset.map(
        lambda x: preprocess_examples_for_model(x, tokenizer), batched=True
    )
    test_dataset_positive = test_dataset.map(
        lambda x: preprocess_examples_for_model(x, tokenizer), batched=True
    )
    # ---- Preprocess negative examples ----
    train_dataset_negative = train_dataset.map(
        preprocess_negative_examples_for_model,
        batched=True,
        remove_columns=train_dataset.column_names,
        fn_kwargs={"stmt2id": stmt2id, "tokenizer": tokenizer}
    )
    val_dataset_negative = val_dataset.map(
        preprocess_negative_examples_for_model,
        batched=True,
        remove_columns=val_dataset.column_names,
        fn_kwargs={"stmt2id": stmt2id, "tokenizer": tokenizer}
    )
    test_dataset_negative = test_dataset.map(
        preprocess_negative_examples_for_model,
        batched=True,
        remove_columns=test_dataset.column_names,
        fn_kwargs={"stmt2id": stmt2id, "tokenizer": tokenizer}
    )
    # ---- Sample negative examples ----
    k = 0.08  # Number of negative examples per positive example

    num_positives_train = len(train_dataset_positive)
    num_positives_val = len(val_dataset_positive)
    num_positives_test = len(test_dataset_positive)

    train_dataset_negative_sampled = (train_dataset_negative.shuffle(seed=42).
                                      select(range(min(len(train_dataset_negative), int(k * num_positives_train)))))
    val_dataset_negative_sampled = (val_dataset_negative.shuffle(seed=42).
                                      select(range(min(len(val_dataset_negative), int(k * num_positives_val)))))
    test_dataset_negative_sampled = (test_dataset_negative.shuffle(seed=42).
                                      select(range(min(len(test_dataset_negative), int(k * num_positives_test)))))

    # Shuffle and concatenate positive and negative examples
    train_dataset = concatenate_datasets([train_dataset_positive, train_dataset_negative_sampled])
    val_dataset = concatenate_datasets([val_dataset_positive, val_dataset_negative_sampled])
    test_dataset = concatenate_datasets([test_dataset_positive, test_dataset_negative_sampled])

    # Shuffle the datasets
    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = val_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)

    # ---- Log dataset sizes ----
    from collections import Counter
    label_counts = Counter(train_dataset["labels"])
    print("Label distribution in training data:", label_counts) 

    # ---- Model ----
    model = BertForIndraStmtClassification.from_pretrained_with_labels(
        pretrained_model_name=args.model_name,
        label2id=stmt2id,
        id2label=id2stmt,
    )
    # Resize token embeddings to accommodate <e> and </e>
    model.resize_token_embeddings(len(tokenizer))
    
    # ---- Training ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=15,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir="./logs",
    )

    class_weights = compute_class_weights(train_dataset)
    data_collator = DataCollatorWithEntitySpans(tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
