from datetime import datetime
import argparse
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from prepare_ner_dataset import (generator_function, 
                                 preprocess_examples,
                                 label2id, id2label)



def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions, true_labels = [], []
    for pred, label in zip(predictions, labels):
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                true_predictions.append(p_)
                true_labels.append(l_)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average="macro")
    return {"precision": precision, "recall": recall, "f1": f1}


def main(args):
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ---- Load dataset ----
    dataset = Dataset.from_generator(generator_function(dataset_path, tokenizer))

    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
    train_dataset = split_dataset["train"]
    temp_dataset = split_dataset["test"]
    val_test_split = temp_dataset.train_test_split(test_size=1/3, seed=42)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    train_dataset = train_dataset.map(lambda x: preprocess_examples(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: preprocess_examples(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: preprocess_examples(x, tokenizer), batched=True)

    # ---- Load model ----
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
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
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate the model on the test set
    print("Running final evaluation on the test set...")
    test_metrics = trainer.evaluate(test_dataset)

    # Save test evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"test_eval_results_{timestamp}.txt"
    with open(log_file, "w") as f:
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Test evaluation results saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model")

    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSONL file")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained model")

    args = parser.parse_args()
    main(args)
