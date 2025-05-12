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
        # Collapse members.0, members.1 → members
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

            example = {
                "matches_hash": matches_hash,
                "statement_type": statement_type,
                "annotated_text": annotated_text
            }
            examples.append(example)
    return examples

# ---- Build label maps from dataset ----
def build_label_mappings(examples, tokenizer):
    label_set = set()

    for example in examples:
        clean_text, role_spans = parse_role_spans(example["annotated_text"])
        encoding = tokenizer(
            text=example["statement_type"],
            text_pair=clean_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            padding="max_length",
            add_special_tokens=True
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offsets = encoding["offset_mapping"]
        seq_ids = encoding.sequence_ids()
        relevant_offsets = [
            offsets[i] if sid == 1 else (None, None)
            for i, sid in enumerate(seq_ids)
        ]
        bio_tags = char_to_token_labels(tokens, relevant_offsets, role_spans)
        label_set.update(bio_tags)

    label2id = {label: idx for idx, label in enumerate(sorted(label_set))}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label

# ---- Tokenize for model ----
def preprocess_examples_from_dataset(batch, tokenizer, label2id):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    tokens_list = []
    ner_tags_list = []
    matches_hash_list = []

    for annotated_text, stmt_type, match_id in zip(
        batch["annotated_text"], batch["statement_type"], batch["matches_hash"]
    ):
        clean_text, role_spans = parse_role_spans(annotated_text)

        encoding = tokenizer(
            text=stmt_type,
            text_pair=clean_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            padding="max_length",
            add_special_tokens=True
        )

        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offsets = encoding["offset_mapping"]
        seq_ids = encoding.sequence_ids()

        relevant_offsets = [
            offsets[i] if sid == 1 else (None, None)
            for i, sid in enumerate(seq_ids)
        ]
        bio_tags = char_to_token_labels(tokens, relevant_offsets, role_spans)

        label_ids = [
            -100 if offset == (None, None) else label2id[bio_tag]
            for offset, bio_tag in zip(relevant_offsets, bio_tags)
        ]

        input_ids_list.append(encoding["input_ids"])
        attention_mask_list.append(encoding["attention_mask"])
        labels_list.append(label_ids)
        tokens_list.append(tokens)
        ner_tags_list.append(bio_tags)
        matches_hash_list.append(match_id)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
        "tokens": tokens_list,
        "ner_tags": ner_tags_list,
        "matches_hash": matches_hash_list
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
        padding="max_length",
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
