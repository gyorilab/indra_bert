import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

def load_tsv_dataset(tsv_path: Path) -> List[Dict[str, Any]]:
    """Load examples from TSV file."""
    examples = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            examples.append({
                'doc_id': row['doc_id'],
                'event_type': row['event_type'],
                'event_id': row['event_id'],
                'text': row['text'],
                'annotated_text': row['annotated_text'],
                'entity_text': row['entity_text'],
                'entity_start': int(row['entity_start']),
                'entity_end': int(row['entity_end']),
                'trigger_text': row['trigger_text'],
                'trigger_start': int(row['trigger_start']),
                'trigger_end': int(row['trigger_end']),
                'role': row['role'],  # Head or Tail
            })
    return examples


def extract_entity_from_annotated(annotated_text: str) -> Tuple[str, int, int]:
    """
    Extract entity text and its position from annotated text.
    Returns (entity_text, start_pos, end_pos) in the original text (without tags).
    """
    match = re.search(r'<e>(.*?)</e>', annotated_text)
    if not match:
        return None, -1, -1
    
    entity_text = match.group(1)
    # Calculate position in original text (accounting for tags before this position)
    tag_start = match.start()
    # Count characters before this tag (excluding tag characters)
    chars_before = len(re.sub(r'<[^>]+>', '', annotated_text[:tag_start]))
    start_pos = chars_before
    end_pos = start_pos + len(entity_text)
    
    return entity_text, start_pos, end_pos


def char_to_token_labels(tokens, offset_mapping, trigger_start: int, trigger_end: int, role: str, special_tokens=None):
    """
    Create BIO labels for trigger span with role information.
    
    Labels: B-trigger-Head, I-trigger-Head, B-trigger-Tail, I-trigger-Tail, O
    """
    if special_tokens is None:
        special_tokens = {"<e>", "</e>"}
    
    labels = ["O"] * len(tokens)
    role_label = role  # "Head" or "Tail"
    
    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        # Skip special tokens
        if tokens[i] in special_tokens:
            continue
        if tok_start is None or tok_end is None:
            continue
        if tok_end <= trigger_start or tok_start >= trigger_end:
            continue
        
        # Token overlaps with trigger span
        if tok_start == trigger_start:
            labels[i] = f"B-trigger-{role_label}"
        else:
            labels[i] = f"I-trigger-{role_label}"
    
    return labels


def build_label_mappings(examples):
    """Build label2id and id2label mappings from examples."""
    label_set = {"O"}  # Always include O
    
    for example in examples:
        role = example['role']
        label_set.add(f"B-trigger-{role}")
        label_set.add(f"I-trigger-{role}")
    
    label2id = {label: idx for idx, label in enumerate(sorted(label_set))}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def preprocess_examples(example: Dict[str, Any], tokenizer, label2id: Dict[str, int]) -> Dict[str, Any]:
    """
    Preprocess a single example for training.
    
    Input: text with <e>entity</e> tagged
    Output: tokenized input with BIO labels for trigger span and role
    """
    # Use annotated_text which has <e>entity</e> tags
    annotated_text = example['annotated_text']
    
    # Remove <trigger> tags from annotated_text for input (we want to predict triggers)
    # Keep only <e> tags
    input_text = re.sub(r'</?trigger>', '', annotated_text)
    
    # Get original text (without any tags) for span calculations
    original_text = example['text']
    
    # Tokenize the input_text (with <e> tags)
    encoding = tokenizer(
        text=input_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        padding=False,
        add_special_tokens=True,
    )
    
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offset_mapping = encoding["offset_mapping"]
    
    # Get trigger span from original text coordinates
    trigger_start = example['trigger_start']
    trigger_end = example['trigger_end']
    role = example['role']
    
    # Map trigger positions from original_text to input_text coordinates
    # Since input_text has <e> tags but original_text doesn't, we need to adjust
    # Build mapping: for each position in original_text, find corresponding position in input_text
    orig_to_input_map = {}
    orig_idx = 0
    input_idx = 0
    
    # Build character position mapping by skipping tags in input_text
    while orig_idx < len(original_text) and input_idx < len(input_text):
        if input_text[input_idx] == '<':
            # Skip tag
            while input_idx < len(input_text) and input_text[input_idx] != '>':
                input_idx += 1
            if input_idx < len(input_text):
                input_idx += 1
        else:
            # This character corresponds to original_text[orig_idx]
            orig_to_input_map[orig_idx] = input_idx
            orig_idx += 1
            input_idx += 1
    
    # Map trigger positions
    mapped_trigger_start = orig_to_input_map.get(trigger_start, trigger_start)
    mapped_trigger_end = orig_to_input_map.get(trigger_end - 1, trigger_end - 1) + 1 if (trigger_end - 1) in orig_to_input_map else trigger_end
    
    # Create BIO labels using mapped positions
    bio_tags = char_to_token_labels(tokens, offset_mapping, mapped_trigger_start, mapped_trigger_end, role)
    
    # Convert labels to IDs
    label_ids = [label2id.get(tag, label2id["O"]) for tag in bio_tags]
    
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": label_ids,
        "offset_mapping": offset_mapping,
        "tokens": tokens,
        "text": original_text,  # Store original text for evaluation
        "annotated_text": input_text,  # Store input text with <e> tags
        "entity_text": example['entity_text'],
        "trigger_text": example['trigger_text'],
        "role": role,
    }

