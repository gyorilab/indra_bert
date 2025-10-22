# class_weights_utils.py

from collections import Counter


def compute_class_weights(dataset, label_key="labels"):
    """
    Compute inverse frequency class weights for handling class imbalance.
    Only computes weights for positive classes (excludes -1 labels).
    
    Args:
        dataset: HuggingFace dataset with labels
        label_key: Key to access labels in dataset items
    
    Returns:
        List of class weights (higher weights for rarer classes)
    """
    labels = [item[label_key] for item in dataset]
    label_counts = Counter(labels)
    
    # Remove -1 (negative examples) from class weight computation
    # Gate 2 only trains on positive examples, so we only need weights for those
    if -1 in label_counts:
        del label_counts[-1]
    
    if not label_counts:
        return []
    
    # Find max positive class ID
    num_classes = max(label_counts.keys()) + 1
    total = sum(label_counts.values())

    class_weights = []
    for i in range(num_classes):
        freq = label_counts.get(i, 1)  # Use 1 for unseen classes to avoid zero division
        weight = total / (num_classes * freq)
        class_weights.append(weight)

    return class_weights


def compute_gate1_weights(dataset, label_key="labels"):
    """
    Compute class weights for Gate 1 binary classification.
    
    Args:
        dataset: HuggingFace dataset with labels
        label_key: Key to access labels in dataset items
    
    Returns:
        List of 2 weights: [weight_for_no_relation, weight_for_has_relation]
    """
    labels = [item[label_key] for item in dataset]
    
    # Convert to gate1 labels: -1 becomes 0 (no_relation), others become 1 (has_relation)
    gate1_labels = [0 if label == -1 else 1 for label in labels]
    
    label_counts = Counter(gate1_labels)
    total = sum(label_counts.values())
    
    # Compute inverse frequency weights for binary classification
    weight_no_relation = total / (2 * label_counts.get(0, 1))
    weight_has_relation = total / (2 * label_counts.get(1, 1))
    
    return [weight_no_relation, weight_has_relation]
