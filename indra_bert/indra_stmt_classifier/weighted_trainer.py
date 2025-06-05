# weighted_trainer.py

import torch
from transformers import Trainer
from collections import Counter
import numpy as np


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        if self.class_weights is not None:
            weight_tensor = torch.tensor(self.class_weights, device=logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(dataset, label_key="stmt_label_id"):
    labels = [item[label_key] for item in dataset]
    label_counts = Counter(labels)
    num_classes = max(label_counts.keys()) + 1
    total = sum(label_counts.values())

    class_weights = []
    for i in range(num_classes):
        freq = label_counts.get(i, 1)  # Use 1 for unseen classes to avoid zero division
        weight = total / (num_classes * freq)
        class_weights.append(weight)

    return class_weights
