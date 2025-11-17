"""
Preprocessing for negation detection dataset.
Loads JSON dataset and creates BIO labels for cues and scopes.
"""
import json
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100


def load_negation_dataset(json_path: Path) -> List[Dict[str, Any]]:
    """Load negation examples from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_label_mappings():
    """Build label2id and id2label mappings for cues and scopes."""
    cue_labels = ["O", "B-cue", "I-cue"]
    scope_labels = ["O", "B-scope", "I-scope"]
    
    cue_label2id = {label: idx for idx, label in enumerate(cue_labels)}
    cue_id2label = {idx: label for label, idx in cue_label2id.items()}
    
    scope_label2id = {label: idx for idx, label in enumerate(scope_labels)}
    scope_id2label = {idx: label for label, idx in scope_label2id.items()}
    
    return (
        cue_label2id, cue_id2label,
        scope_label2id, scope_id2label
    )


def char_to_token_labels(
    tokens: List[str],
    offset_mapping: List[tuple],
    span_start: int,
    span_end: int,
    label_type: str = "cue",  # "cue" or "scope"
) -> List[str]:
    """
    Create BIO labels for a span (cue or scope).
    
    Args:
        tokens: List of token strings
        offset_mapping: List of (start, end) tuples for each token
        span_start: Character start position of the span
        span_end: Character end position of the span
        label_type: "cue" or "scope" to determine label prefix
    
    Returns:
        List of BIO labels (O, B-{label_type}, I-{label_type})
    """
    labels = ["O"] * len(tokens)
    B_label = f"B-{label_type}"
    I_label = f"I-{label_type}"
    
    first_token_found = False
    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start is None or tok_end is None:
            continue
        
        # Check if token overlaps with span
        if tok_end <= span_start or tok_start >= span_end:
            continue
        
        # Token overlaps with span
        if not first_token_found:
            labels[i] = B_label
            first_token_found = True
        else:
            labels[i] = I_label
    
    return labels


def tokenize_negation_dataset(
    examples: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    cue_label2id: Dict[str, int],
    scope_label2id: Dict[str, int],
) -> Dataset:
    """
    Tokenize negation examples and create token-level labels for cues and scopes.
    
    Args:
        examples: List of examples with 'text' and 'negations' keys
        tokenizer: Tokenizer instance
        cue_label2id: Mapping from cue labels to IDs
        scope_label2id: Mapping from scope labels to IDs
    
    Returns:
        Dataset with tokenized inputs and labels
    """
    processed_examples = []
    
    for example in examples:
        text = example['text']
        negations = example.get('negations', [])
        
        # Tokenize text
        encoding = tokenizer(
            text=text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            padding=False,
            add_special_tokens=True,
        )
        
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offset_mapping = encoding["offset_mapping"]
        
        # Initialize labels as all O
        cue_labels = ["O"] * len(tokens)
        scope_labels = ["O"] * len(tokens)
        
        # Sort negations by scope start position to process outer scopes first
        # This helps preserve outer scopes when they contain inner scopes
        sorted_negations = sorted(
            negations,
            key=lambda n: (n.get('scope', {}).get('start', 0), -n.get('scope', {}).get('end', 0))
        )
        
        # Process each negation
        for negation in sorted_negations:
            cue = negation.get('cue', {})
            scope = negation.get('scope', {})
            
            cue_start = cue.get('start')
            cue_end = cue.get('end')
            scope_start = scope.get('start')
            scope_end = scope.get('end')
            
            if cue_start is not None and cue_end is not None:
                # Create cue labels
                cue_bio = char_to_token_labels(
                    tokens, offset_mapping, cue_start, cue_end, label_type="cue"
                )
                # Merge with existing labels (prioritize non-O labels)
                for i, label in enumerate(cue_bio):
                    if label != "O":
                        cue_labels[i] = label
            
            if scope_start is not None and scope_end is not None:
                # Create scope labels
                scope_bio = char_to_token_labels(
                    tokens, offset_mapping, scope_start, scope_end, label_type="scope"
                )
                # For overlapping scopes, we want to preserve the outer scope
                # Check if this scope overlaps with an existing scope
                has_overlap = False
                for i, existing_label in enumerate(scope_labels):
                    if existing_label != "O" and scope_bio[i] != "O":
                        # Check if this is an inner scope (starts later) within an outer scope
                        # In that case, we still want to mark both, but prioritize outer scope boundaries
                        has_overlap = True
                        break
                
                # Merge labels: if there's overlap and this scope starts later (inner scope),
                # we still mark it but don't overwrite the outer scope's B tag
                for i, label in enumerate(scope_bio):
                    if label != "O":
                        # If this is a B tag and there's already a scope label here,
                        # it might be the start of an inner scope - mark it as I-scope instead
                        # to preserve the outer scope's B tag
                        if label == "B-scope" and scope_labels[i] != "O":
                            # This is an inner scope starting within an outer scope
                            # Keep the outer scope's label (likely B-scope)
                            pass  # Don't overwrite
                        else:
                            # No conflict or this is I-scope, safe to set
                            scope_labels[i] = label
        
        # Convert labels to IDs
        cue_label_ids = [cue_label2id.get(label, 0) for label in cue_labels]
        scope_label_ids = [scope_label2id.get(label, 0) for label in scope_labels]
        
        example_dict = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "cue_labels": cue_label_ids,
            "scope_labels": scope_label_ids,
            "text": text,  # Store original text for metrics
        }
        
        if "token_type_ids" in encoding:
            example_dict["token_type_ids"] = encoding["token_type_ids"]
        
        processed_examples.append(example_dict)
    
    return Dataset.from_list(processed_examples)

