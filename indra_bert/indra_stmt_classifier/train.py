import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import Counter
import torch

from datasets import Dataset, concatenate_datasets, DatasetDict, load_from_disk
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding, Trainer
from sklearn.metrics import precision_recall_fscore_support

from .bert_classification_head import TwoGatedClassifier
from .preprocess import (
    load_and_preprocess_raw_data,
    preprocess_examples_for_model,
    preprocess_negative_examples_for_model
)
from .class_weights_utils import compute_class_weights, compute_gate1_weights


class TwoGatedDataCollator:
    def __init__(self, tokenizer, class_weights=None, gate1_weights=None):
        self.tokenizer = tokenizer
        self.default_collator = DataCollatorWithPadding(tokenizer)
        self.class_weights = class_weights
        self.gate1_weights = gate1_weights
        self.print_counter = 0

    def __call__(self, features):
        batch = self.default_collator(features)
        
        # Generate Gate 1 labels dynamically
        gate1_labels = []
        for label in batch['labels']:
            if label == -1:  # Negative example
                gate1_labels.append(0)  # no_relation
            else:
                gate1_labels.append(1)  # has_relation
        
        batch['gate1_labels'] = torch.tensor(gate1_labels, dtype=torch.long)
        
        # Add class weights to batch if available
        if self.class_weights is not None:
            batch['class_weights'] = self.class_weights
        
        # Add gate1 weights to batch if available
        if self.gate1_weights is not None:
            batch['gate1_weights'] = self.gate1_weights
        
        if self.print_counter < 3:
            print("Example batch item:", self.tokenizer.decode(batch['input_ids'][0]))
            print("Gate2 label:", batch['labels'][0].item())
            print("Gate1 label:", batch['gate1_labels'][0].item())
            self.print_counter += 1
        
        return batch


def compute_metrics(p):
    """
    Compute metrics for two-gated hierarchical classification.
    Now uses the clean tuple format: (gate1_logits, gate2_logits)
    """
    import torch
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    # p.predictions should now be just the main 'logits' tuple: (gate1_logits, gate2_logits)
    if isinstance(p.predictions, tuple) and len(p.predictions) == 2:
        gate1_logits = torch.tensor(p.predictions[0])  # Gate 1 logits
        gate2_logits = torch.tensor(p.predictions[1])  # Gate 2 logits
    else:
        raise ValueError(f"Expected predictions to be tuple (gate1_logits, gate2_logits), but got {type(p.predictions)} with {len(p.predictions) if hasattr(p.predictions, '__len__') else 'unknown'} elements")
    
    if isinstance(p.label_ids, tuple):
        labels = p.label_ids[0]
    else:
        labels = p.label_ids
    
    # Apply the same logic as model.predict()
    gate1_probs = torch.softmax(gate1_logits, dim=-1)
    gate2_probs = torch.softmax(gate2_logits, dim=-1) 
    has_relation_prob = gate1_probs[:, 1]
    
    # Use fixed threshold for evaluation (we can't access learned threshold here)
    gate1_threshold = 0.5
    
    predictions = []
    for i in range(len(has_relation_prob)):
        if has_relation_prob[i] > gate1_threshold:
            relation_type_idx = torch.argmax(gate2_probs[i])
            predictions.append(relation_type_idx.item())
        else:
            predictions.append(-1)
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Treat -1 as just another class for evaluation
    accuracy = accuracy_score(labels, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predictions, average="micro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro, 
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def evaluate_model_with_predict(model, dataset, tokenizer):
    """
    Evaluate the model using its predict method with the learned threshold.
    This gives us the true performance of the model as it will be used in inference.
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from .preprocess import preprocess_for_inference
    import torch
    
    model.eval()
    predictions = []
    true_labels = []
    
    print(f"Running evaluation using model.predict() with learned threshold: {model.gate1_threshold.item():.4f}")
    
    for i, example in enumerate(dataset):
        # Get the original annotated text
        annotated_text = example.get('annotated_text')
        if annotated_text is None:
            # Skip if no annotated text (shouldn't happen but safety check)
            continue
            
        true_label = example['labels']
        
        # Use the model's predict method
        enc = preprocess_for_inference(annotated_text, tokenizer)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        
        with torch.no_grad():
            result = model.predict(input_ids, attention_mask)
            pred_label = result['predictions'][0]
            
        predictions.append(pred_label)
        true_labels.append(true_label)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics treating -1 as just another class
    accuracy = accuracy_score(true_labels, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels, predictions, average="micro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted", zero_division=0
    )
    
    return {
        "predict_accuracy": accuracy,
        "predict_precision_macro": precision_macro,
        "predict_recall_macro": recall_macro,
        "predict_f1_macro": f1_macro,
        "predict_precision_micro": precision_micro,
        "predict_recall_micro": recall_micro,
        "predict_f1_micro": f1_micro,
        "predict_precision_weighted": precision_weighted,
        "predict_recall_weighted": f1_weighted,
        "predict_f1_weighted": f1_weighted,
        "learned_threshold": model.gate1_threshold.item(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--use_cached_dataset", action="store_true")
    parser.add_argument("--max_negatives_per_positive", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dataset_path = output_dir / "cached_dataset"
    cache_stmt_path = cache_dataset_path / "stmt2id.npy"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<e>', '</e>']})

    if args.use_cached_dataset and cache_dataset_path.exists() and cache_stmt_path.exists():
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
            fn_kwargs={"tokenizer": tokenizer}
        )
        val_neg = val_dataset.map(
            preprocess_negative_examples_for_model,
            batched=True,
            remove_columns=val_dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer}
        )
        test_neg = test_dataset.map(
            preprocess_negative_examples_for_model,
            batched=True,
            remove_columns=test_dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer}
        )

        # Sample negatives (k = 1)
        k = args.max_negatives_per_positive
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
    training_config = vars(args).copy()  # Convert Namespace -> dict
    training_config["time_created"] = datetime.now().strftime("%Y-%m-%d")
    model = TwoGatedClassifier.from_pretrained_with_labels(
        pretrained_model_name=args.model_name,
        label2id=stmt2id,
        id2label=id2stmt,
        training_config=training_config
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        print("Resizing token embeddings to match tokenizer size...")
        print("Old embedding size:", model.get_input_embeddings().num_embeddings)
        print("New embedding size:", len(tokenizer))
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
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            save_total_limit=1,
            logging_dir="./logs",
    )

    # Compute class weights for both gates
    class_weights = compute_class_weights(train_dataset)  # For Gate 2 (multi-class)
    gate1_weights = compute_gate1_weights(train_dataset)  # For Gate 1 (binary)
    
    print(f"Gate 2 class weights: {class_weights}")
    print(f"Gate 1 class weights: {gate1_weights}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=TwoGatedDataCollator(tokenizer, class_weights, gate1_weights),
        compute_metrics=compute_metrics,
    )

    # ---- Train and Evaluate ----
    trainer.train()

    # ---- Final evaluation using model.predict() with learned threshold ----
    print("Running final evaluation on test set using model.predict()...")
    test_metrics = evaluate_model_with_predict(model, test_dataset, tokenizer)
    
    # Also run standard trainer evaluation for comparison
    print("Running standard trainer evaluation for comparison...")
    trainer_test_metrics = trainer.evaluate(test_dataset)
    
    # Combine both sets of metrics
    combined_metrics = {**trainer_test_metrics, **test_metrics}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"test_eval_results_{timestamp}.txt"
    with open(log_file, "w") as f:
        f.write(f"Name: {args.model_name}\n")
        f.write(f"Label distribution in training data: {label_counts}\n")
        f.write("\n")
        for key, value in combined_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Test evaluation results saved to {log_file}")


if __name__ == "__main__":
    main()
