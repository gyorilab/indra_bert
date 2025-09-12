# class_weights_utils.py

from collections import Counter


def compute_class_weights(dataset, label_key="stmt_label_id"):
    """
    Compute inverse frequency class weights for handling class imbalance.
    
    Args:
        dataset: HuggingFace dataset with labels
        label_key: Key to access labels in dataset items
    
    Returns:
        List of class weights (higher weights for rarer classes)
    """
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
