import sys
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from models.agent_role_classifier import load_agent_role_classifier
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    true_preds = []
    true_labels = []
    for p, l in zip(preds, labels):
        for p_, l_ in zip(p, l):
            if l_ != -100:
                true_preds.append(p_)
                true_labels.append(l_)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average="macro")
    return {"precision": precision, "recall": recall, "f1": f1}

def main(dataset_path, model_name, output_dir):
    dataset = Dataset.from_json(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):
        return tokenizer(example['text'], truncation=True, padding="max_length")

    dataset = dataset.map(tokenize)

    split = dataset.train_test_split(test_size=0.2)
    train_dataset = split['train']
    eval_dataset = split['test']

    model = load_agent_role_classifier(model_name, num_labels=len(dataset.features['labels'].feature.names))

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    model_name = sys.argv[2]
    output_dir = sys.argv[3]
    main(dataset_path, model_name, output_dir)
