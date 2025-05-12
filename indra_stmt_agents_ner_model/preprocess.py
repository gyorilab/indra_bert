import json
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset

# ---- Parse annotated spans ----
def parse_role_spans(text):
    spans = []
    clean_text = ""
    last_end = 0

    for match in re.finditer(r"<([^<>]+?)>(.*?)</\1>", text):
        role = match.group(1)
        span_text = match.group(2)
        start, end = match.span()

        clean_text += text[last_end:start]
        span_start = len(clean_text)
        clean_text += span_text
        span_end = len(clean_text)

        spans.append((span_start, span_end, role))
        last_end = end

    clean_text += text[last_end:]
    return clean_text, spans

# ---- Assign BIO tags ----
def char_to_token_labels(tokens, token_offsets, role_spans):
    labels = ["O"] * len(tokens)
    for start_char, end_char, role in role_spans:
        collapsed_role = role.split('.')[0] if role.startswith("members.") else role
        for i, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_start is None or tok_end is None:
                continue
            if tok_end <= start_char or tok_start >= end_char:
                continue
            if tok_start == start_char:
                labels[i] = f"B-{collapsed_role}"
            else:
                labels[i] = f"I-{collapsed_role}"
    return labels

# ---- Reassign member indices after prediction ----
def reassign_member_indices(predicted_tags):
    reassigned = []
    current_index = -1
    for tag in predicted_tags:
        if tag == "B-members":
            current_index += 1
            reassigned.append(f"B-members.{current_index}")
        elif tag == "I-members":
            if current_index == -1:
                reassigned.append("I-members.0")
            else:
                reassigned.append(f"I-members.{current_index}")
        else:
            reassigned.append(tag)
    return reassigned

# ---- Load raw examples ----
def load_and_preprocess_from_raw_data(input_path):
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for _, line in enumerate(tqdm(f, desc="Loading and preprocessing")):
            obj = json.loads(line)
            matches_hash = obj["matches_hash"]
            statement_type = obj["statement"]["type"]
            annotated_text = obj["annotated_text"]
            examples.append({
                "matches_hash": matches_hash,
                "statement_type": statement_type,
                "annotated_text": annotated_text
            })
    return examples

# ---- Build label maps from dataset ----
def build_label_mappings(examples):
    label_set = set()
    for example in examples:
        clean_text, role_spans = parse_role_spans(example["annotated_text"])
        tokens = clean_text.split()
        char_spans = []
        current = 0
        for tok in tokens:
            start = clean_text.find(tok, current)
            end = start + len(tok)
            char_spans.append((start, end))
            current = end
        bio_tags = char_to_token_labels(tokens, char_spans, role_spans)
        label_set.update(bio_tags)

    label2id = {label: idx for idx, label in enumerate(sorted(label_set))}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label

# ---- Tokenize for model ----
def preprocess_examples_from_dataset(example, tokenizer, label2id):
    annotated_text = example["annotated_text"]
    stmt_type = example["statement_type"]
    match_id = example["matches_hash"]

    # Remove tags and get spans
    clean_text, role_spans = parse_role_spans(annotated_text)

    # Tokenize full pair
    encoding = tokenizer(
        text=stmt_type,
        text_pair=clean_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        padding=False, # Handle padding in collator in the training
        add_special_tokens=True,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]
    seq_ids = encoding.sequence_ids()

    # Mask non-clean_text part using sequence_ids
    relevant_offsets = [
        offsets[i] if sid == 1 else (None, None)
        for i, sid in enumerate(seq_ids)
    ]

    # Create BIO tags
    bio_tags = char_to_token_labels(tokens, relevant_offsets, role_spans)

    # Assign label ids
    label_ids = [
        -100 if offset == (None, None) else label2id[bio_tag]
        for offset, bio_tag in zip(relevant_offsets, bio_tags)
    ]

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": label_ids,
        "tokens": tokens,
        "ner_tags": bio_tags,
        "matches_hash": match_id
    }

# ---- Preprocess for inference ----
def preprocess_for_inference(stmt_type, text, tokenizer):
    encoding = tokenizer(
        text=stmt_type,
        text_pair=text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="longest",
        add_special_tokens=True
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    offsets = encoding["offset_mapping"][0].tolist()
    sequence_ids = encoding.sequence_ids()

    return {
        "tokens": tokens,
        "offsets": offsets,
        "sequence_ids": sequence_ids,
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"]
    }
