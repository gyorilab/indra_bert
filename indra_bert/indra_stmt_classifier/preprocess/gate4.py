"""
Preprocessing for Gate4: HRT predicate detection (token classification).
"""
import csv
import re
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
from transformers import PreTrainedTokenizer

from .shared import IGNORE_INDEX


def load_hrt_dataset(tsv_path: Path) -> List[Dict[str, Any]]:
    """Load HRT examples from TSV file."""
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
                'head_text': row['head_text'],
                'head_start': int(row['head_start']),
                'head_end': int(row['head_end']),
                'tail_text': row['tail_text'],
                'tail_start': int(row['tail_start']),
                'tail_end': int(row['tail_end']),
                'trigger_text': row['trigger_text'],
                'trigger_start': int(row['trigger_start']),
                'trigger_end': int(row['trigger_end']),
            })
    return examples


def build_gate4_label_mapping():
    """Build label2id and id2label mappings for HRT format (simpler labels)."""
    labels = ["O", "B-trigger", "I-trigger"]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def char_to_token_labels(tokens, offset_mapping, trigger_start: int, trigger_end: int, special_tokens=None):
    """
    Create BIO labels for trigger span (no role information).
    
    Labels: B-trigger, I-trigger, O
    """
    if special_tokens is None:
        special_tokens = {"<e>", "</e>"}
    
    labels = ["O"] * len(tokens)
    
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
            labels[i] = "B-trigger"
        else:
            labels[i] = "I-trigger"
    
    return labels


def tokenize_hrt_dataset(examples: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, label2id: Dict[str, int]) -> Dataset:
    """
    Tokenize HRT examples for training.
    
    Input: text with <e>head</e> and <e>tail</e> tagged
    Output: tokenized input with BIO labels for trigger span (no role)
    """
    processed_examples = []
    
    for example in examples:
        # Use annotated_text which has <e>head</e>, <e>tail</e>, and <trigger> tags
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
        
        # Get trigger span from original text coordinates (character positions in untagged text)
        trigger_start = example['trigger_start']
        trigger_end = example['trigger_end']
        
        # Map trigger positions from original_text to input_text coordinates
        # BIO tags are assigned based on token positions in the tagged input text (with <e> tags),
        # but the gold trigger spans are given in original_text coordinates (without tags).
        # So we need to map: original_text position -> input_text position -> token position -> BIO label
        # Build mapping: for each position in original_text, find corresponding position in input_text
        orig_to_input_map = {}
        orig_idx = 0
        input_idx = 0
        
        # Build character position mapping by skipping tags in input_text
        while orig_idx < len(original_text) and input_idx < len(input_text):
            if input_text[input_idx] == '<':
                # Skip tag characters
                while input_idx < len(input_text) and input_text[input_idx] != '>':
                    input_idx += 1
                if input_idx < len(input_text):
                    input_idx += 1
            else:
                # Map: character position in original_text -> character position in input_text (with tags)
                orig_to_input_map[orig_idx] = input_idx
                orig_idx += 1
                input_idx += 1
        
        # Map trigger positions from original_text to input_text (with tags)
        mapped_trigger_start = orig_to_input_map.get(trigger_start, trigger_start)
        # For end position, try trigger_end-1 first (since trigger_end is exclusive)
        if (trigger_end - 1) in orig_to_input_map:
            mapped_trigger_end = orig_to_input_map[trigger_end - 1] + 1
        elif trigger_end in orig_to_input_map:
            mapped_trigger_end = orig_to_input_map[trigger_end]
        else:
            mapped_trigger_end = trigger_end  # Fallback
        
        # Create BIO labels using mapped positions in tagged text (no role)
        # These labels will be used during training, and during inference we'll map back to original_text
        bio_tags = char_to_token_labels(tokens, offset_mapping, mapped_trigger_start, mapped_trigger_end)
        
        # Convert labels to IDs
        label_ids = [label2id.get(tag, label2id["O"]) for tag in bio_tags]
        
        example_dict = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "gate4_labels": label_ids,
            "gate1_labels": IGNORE_INDEX,  # Not used for HRT dataset
            "gate2_labels": IGNORE_INDEX,  # Not used for HRT dataset
            "gate3_labels": IGNORE_INDEX,  # Not used for HRT dataset
        }
        
        # Include token_type_ids if present (some tokenizers return it)
        if "token_type_ids" in encoding:
            example_dict["token_type_ids"] = encoding["token_type_ids"]
        
        processed_examples.append(example_dict)
    
    return Dataset.from_list(processed_examples)

