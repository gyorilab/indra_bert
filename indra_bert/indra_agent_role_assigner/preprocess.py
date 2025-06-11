import json
import re
from pathlib import Path
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import torch

# ---- Parse annotated spans and reformat to <e>...</e> ----
def parse_and_generalize_tags(text):
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

        spans.append({
            "start": span_start, 
            "end": span_end, 
            "entity_text": span_text,
            "role": role
        })
        last_end = end

    clean_text += text[last_end:]

    # Insert generic <e> tags
    entity_annotated_text = clean_text
    for span in sorted(spans, key=lambda x: x["start"], reverse=True):
        entity_annotated_text = entity_annotated_text[:span["end"]] + "</e>" + entity_annotated_text[span["end"]:]
        entity_annotated_text = entity_annotated_text[:span["start"]] + "<e>" + entity_annotated_text[span["start"]:]

    return clean_text, entity_annotated_text, spans

# ---- Assign BIO tags only to entity tokens ----
def char_to_token_labels(tokens, token_offsets, role_spans):
    labels = ["O"] * len(tokens)

    for span_dict in role_spans:
        start_char = span_dict["start"]
        end_char = span_dict["end"]
        role = span_dict["role"]
        collapsed_role = role.split('.')[0] if role.startswith("members.") else role

        inside_span = False
        for i, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_start is None or tok_end is None:
                continue
            if tok_end <= start_char or tok_start >= end_char:
                continue  # Token is outside span

            # Token overlaps with entity span
            if not inside_span:
                labels[i] = f"B-{collapsed_role}"
                inside_span = True
            else:
                labels[i] = f"I-{collapsed_role}"

    return labels

# ---- Load raw examples ----
def load_and_preprocess_from_raw_data(input_path):
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for _, line in enumerate(tqdm(f, desc="Loading and preprocessing")):
            obj = json.loads(line)
            clean_text, entity_annotated_text, spans = parse_and_generalize_tags(obj["annotated_text"])
            examples.append({
                "matches_hash": obj["matches_hash"],
                "statement_type": obj["statement"]["type"],
                "agent_role_annotated_text": obj["annotated_text"],
                "entity_annotated_text": entity_annotated_text,
                "clean_text": clean_text,
                "role_spans": spans
            })
    return examples

# ---- Tokenizer wrapper that fixes offset mapping for special tokens ----
class SpecialTokenOffsetFixTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizer, special_tokens={"<e>", "</e>"}):
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.tokenizer.add_special_tokens({"additional_special_tokens": list(self.special_tokens)})

    def __call__(self, *args, return_offsets_mapping=False, **kwargs):
        encoding = self.tokenizer(*args, return_offsets_mapping=True, **kwargs)

        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        is_batched = isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2

        # Convert offsets to list-of-lists for mutation
        if is_batched:
            offsets = [list(map(tuple, o.tolist())) for o in offsets]
        else:
            offsets = list(map(tuple, offsets))

        # Get tokens for checking special tokens
        batch_tokens = (
            [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
            if is_batched else
            self.tokenizer.convert_ids_to_tokens(input_ids)
        )

        # Patch offsets for special tokens
        if is_batched:
            for i, tokens in enumerate(batch_tokens):
                for j, token in enumerate(tokens):
                    if token in self.special_tokens:
                        offsets[i][j] = (None, None)
        else:
            for j, token in enumerate(batch_tokens):
                if token in self.special_tokens:
                    offsets[j] = (None, None)

        # Assign fixed offsets back
        encoding["offset_mapping"] = offsets

        if not return_offsets_mapping:
            del encoding["offset_mapping"]

        return encoding

    def convert_ids_to_tokens(self, *args, **kwargs):
        return self.tokenizer.convert_ids_to_tokens(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

# ---- Map annotated character positions to clean character positions ----
def map_annotated_to_clean(annotated_text, clean_text):
    annotated_to_clean = {}
    clean_idx = 0
    a_idx = 0

    while a_idx < len(annotated_text) and clean_idx < len(clean_text):
        if annotated_text[a_idx:a_idx+3] == '<e>':
            a_idx += 3
            continue
        elif annotated_text[a_idx:a_idx+4] == '</e>':
            a_idx += 4
            continue
        else:
            annotated_to_clean[a_idx] = clean_idx
            a_idx += 1
            clean_idx += 1

    return annotated_to_clean

# ---- Preprocess for model ----
def preprocess_examples_from_dataset(example, special_tokenizer, label2id):
    encoding = special_tokenizer(
        text=example["statement_type"],
        text_pair=example["entity_annotated_text"],
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        padding=False,
        add_special_tokens=True
    )

    tokens = special_tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]
    seq_ids = encoding.sequence_ids()
    relevant_offsets = [offsets[i] if sid == 1 else (None, None) for i, sid in enumerate(seq_ids)]

    # Map offsets to clean text positions
    offset_map = map_annotated_to_clean(example["entity_annotated_text"], example["clean_text"])
    mapped_offsets = []

    for start, end in relevant_offsets:
        if start is None or end is None or (start == 0 and end == 0):
            mapped_offsets.append((None, None))
        else:
            clean_start = offset_map.get(start)
            clean_end = offset_map.get(end - 1)
            if clean_start is not None and clean_end is not None:
                mapped_offsets.append((clean_start, clean_end + 1))
            else:
                mapped_offsets.append((None, None))

    bio_tags = char_to_token_labels(tokens, mapped_offsets, example["role_spans"])
    label_ids = [-100 if offset == (None, None) else label2id[bio] for offset, bio in zip(mapped_offsets, bio_tags)]

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "sequence_ids": seq_ids,
        "labels": label_ids,
        "tokens": tokens,
        "ner_tags": bio_tags,
        "entity_annotated_text": example["entity_annotated_text"],
        "relevant_offsets": relevant_offsets,
        "mapped_offsets": mapped_offsets,
        "role_spans": example["role_spans"]
    }

# ---- Build label maps from dataset ----
def build_label_mappings(examples):
    label_set = set()
    for example in examples:
        clean_text, entity_annotated_text, role_spans = parse_and_generalize_tags(example["agent_role_annotated_text"])
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

# ---- Preprocess for inference ----
def preprocess_for_inference(stmt_type, entity_annotated_text, special_tokenizer, max_length=512):
    encoding = special_tokenizer(
        text=stmt_type,
        text_pair=entity_annotated_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding="longest",
        add_special_tokens=True,
        return_tensors="pt"
    )

    input_ids_list = encoding["input_ids"][0]
    tokens = special_tokenizer.convert_ids_to_tokens(input_ids_list)

    offsets = encoding["offset_mapping"][0]
    seq_ids = encoding.sequence_ids()

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "tokens": tokens,
        "offsets": offsets,
        "sequence_ids": seq_ids
    }

def reassign_member_indices(bio_tags):
    """
    Converts BIO tags like B-members → B-members.0, B-members.1, etc.
    Keeps other labels (e.g., B-enz, B-sub) unchanged.
    """
    new_tags = []
    member_counter = 0
    current_member_open = False

    for tag in bio_tags:
        if tag.startswith("B-members"):
            tag = f"B-members.{member_counter}"
            current_member_open = True
            member_counter += 1
        elif tag.startswith("I-members"):
            if current_member_open:
                tag = f"I-members.{member_counter - 1}"
            else:
                tag = "O"
        else:
            current_member_open = False
        new_tags.append(tag)

    return new_tags

# ---- Preprocess for inference in batch ----
def preprocess_for_inference_batch(stmt_types, entity_annotated_texts, special_tokenizer, max_length=512):
    """
    Batch preprocessing of multiple (stmt_type, entity_annotated_text) pairs.
    Returns:
        Dictionary of input tensors and batch-level metadata
    """
    if not stmt_types or not entity_annotated_texts or len(stmt_types) != len(entity_annotated_texts):
        return {
            "input_ids": torch.empty(0, dtype=torch.long),
            "attention_mask": torch.empty(0, dtype=torch.long),
            "tokens": [],
            "offsets": [],
            "sequence_ids": []
        }

    encodings = special_tokenizer(
        text=stmt_types,
        text_pair=entity_annotated_texts,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding="longest",
        add_special_tokens=True,
        return_tensors="pt"
    )

    batch_tokens = [special_tokenizer.convert_ids_to_tokens(ids) for ids in encodings["input_ids"]]
    batch_offsets = encodings["offset_mapping"]
    batch_seq_ids = [encodings.sequence_ids(i) for i in range(len(stmt_types))]

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "tokens": batch_tokens,
        "offsets": batch_offsets,
        "sequence_ids": batch_seq_ids
    }

