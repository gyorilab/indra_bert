"""
Preprocessing for two-stage negation detection:
1. Cue detection: token classification (BIO)
2. Scope prediction: span-based (start/end positions given a cue)
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer
import numpy as np

IGNORE_INDEX = -100


def load_negation_dataset(json_path: Path) -> List[Dict[str, Any]]:
    """Load negation examples from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_cue_label_mapping():
    """Build label2id and id2label mappings for cues only."""
    cue_labels = ["O", "B-cue", "I-cue"]
    cue_label2id = {label: idx for idx, label in enumerate(cue_labels)}
    cue_id2label = {idx: label for label, idx in cue_label2id.items()}
    return cue_label2id, cue_id2label


def char_to_token_labels(
    tokens: List[str],
    offset_mapping: List[tuple],
    span_start: int,
    span_end: int,
    label_type: str = "cue",
) -> List[str]:
    """Create BIO labels for a span."""
    labels = ["O"] * len(tokens)
    B_label = f"B-{label_type}"
    I_label = f"I-{label_type}"
    
    first_token_found = False
    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start is None or tok_end is None:
            continue
        
        if tok_end <= span_start or tok_start >= span_end:
            continue
        
        if not first_token_found:
            labels[i] = B_label
            first_token_found = True
        else:
            labels[i] = I_label
    
    return labels


def char_to_token_position(
    offset_mapping: List[tuple],
    char_position: int,
) -> int:
    """
    Convert character position to token position.
    Returns the token index that contains the character position.
    """
    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start is None or tok_end is None:
            continue
        if tok_start <= char_position < tok_end:
            return i
    # If not found, return the closest token
    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start is not None and tok_end is not None:
            if char_position < tok_start:
                return max(0, i - 1)
    return len(offset_mapping) - 1


def tokenize_cue_dataset(
    examples: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    cue_label2id: Dict[str, int],
) -> Dataset:
    """
    Stage 1: Tokenize examples and create BIO labels for cues only.
    One example per text (all cues in the text).
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
        
        # Process each negation cue
        for negation in negations:
            cue = negation.get('cue', {})
            cue_start = cue.get('start')
            cue_end = cue.get('end')
            
            if cue_start is not None and cue_end is not None:
                # Create cue labels
                cue_bio = char_to_token_labels(
                    tokens, offset_mapping, cue_start, cue_end, label_type="cue"
                )
                # Merge with existing labels (prioritize non-O labels)
                for i, label in enumerate(cue_bio):
                    if label != "O":
                        cue_labels[i] = label
        
        # Convert labels to IDs
        cue_label_ids = [cue_label2id.get(label, 0) for label in cue_labels]
        
        example_dict = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "cue_labels": cue_label_ids,
            "text": text,  # Store original text
        }
        
        if "token_type_ids" in encoding:
            example_dict["token_type_ids"] = encoding["token_type_ids"]
        
        processed_examples.append(example_dict)
    
    return Dataset.from_list(processed_examples)


def tokenize_scope_dataset(
    examples: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    """
    Stage 2: Create examples for scope prediction.
    One example per cue-scope pair (each cue gets its own example).
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
        
        offset_mapping = encoding["offset_mapping"]
        
        # Create one example per cue-scope pair
        for negation in negations:
            cue = negation.get('cue', {})
            scope = negation.get('scope', {})
            
            cue_start = cue.get('start')
            cue_end = cue.get('end')
            scope_start = scope.get('start')
            scope_end = scope.get('end')
            
            if cue_start is None or cue_end is None:
                continue
            if scope_start is None or scope_end is None:
                continue
            
            # Convert character positions to token positions
            cue_token_start = char_to_token_position(offset_mapping, cue_start)
            scope_token_start = char_to_token_position(offset_mapping, scope_start)
            scope_token_end = char_to_token_position(offset_mapping, scope_end - 1)  # -1 because end is exclusive
            
            # Ensure positions are valid
            seq_len = len(offset_mapping)
            cue_token_start = max(0, min(cue_token_start, seq_len - 1))
            scope_token_start = max(0, min(scope_token_start, seq_len - 1))
            scope_token_end = max(scope_token_start, min(scope_token_end, seq_len - 1))
            
            example_dict = {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "cue_position": cue_token_start,  # Token index of cue start
                "scope_start_position": scope_token_start,  # Token index of scope start
                "scope_end_position": scope_token_end,      # Token index of scope end
                "text": text,  # Store original text
            }
            
            if "token_type_ids" in encoding:
                example_dict["token_type_ids"] = encoding["token_type_ids"]
            
            processed_examples.append(example_dict)
    
    return Dataset.from_list(processed_examples)

