import json

# ---- Assign BIO tags ----
def char_to_token_labels(tokens, offset_mapping, entity_spans):
    labels = ["O"] * len(tokens)
    for span_dict in entity_spans:
        start_char = span_dict["start"]
        end_char = span_dict["end"]
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start is None or tok_end is None:
                continue
            if tok_end <= start_char or tok_start >= end_char:
                continue
            if tok_start == start_char:
                labels[i] = f"B-entity"
            else:
                labels[i] = f"I-entity"
    return labels

# ---- Load raw examples ----
def load_and_preprocess_from_raw_data(input_path):
    raw_examples = json.load(open(input_path, "r"))
    examples = []
    for raw_example in raw_examples:
        example = {}
        example["pmid"] = raw_example["pmid"]
        example["text"] = raw_example["text"]
        example["annotated_text"] = raw_example["annotated_text"]
        all_entity_spans = []
        for entity in raw_example["entities"]:
            entity_text = entity["text"]
            same_entity_spans = [location for location in entity["locations"]]
            for location in same_entity_spans:
                location["entity_text"] = entity_text
            same_entity_spans = [
                {"start": span['start'], "end": span['end'], "text": span['entity_text']}
                for span in same_entity_spans
            ]
            all_entity_spans.extend(same_entity_spans)
        example["entity_spans"] = all_entity_spans
        examples.append(example)
    return examples

# ---- Build label maps from dataset ----
def build_label_mappings(examples):
    label_set = set()
    for example in examples:
        text = example["text"]
        entity_spans = example["entity_spans"]
        tokens = text.split()

        offset_mapping = []
        current = 0
        for tok in tokens:
            start = text.find(tok, current)
            end = start + len(tok)
            offset_mapping.append((start, end))
            current = end

        bio_tags = char_to_token_labels(tokens, offset_mapping, entity_spans)
        label_set.update(bio_tags)

    label2id = {label: idx for idx, label in enumerate(sorted(label_set))}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


# ---- Preprocess examples ----
def preprocess_examples(example, tokenizer, label2id):

    encoding = tokenizer(
        text=example["text"],
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        padding=False, # Handle padding in collator in the training
        add_special_tokens=True,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offset_mapping = encoding["offset_mapping"]

    # Create BIO tags
    bio_tags = char_to_token_labels(tokens, offset_mapping, example["entity_spans"])

    # Assign label ids
    label_ids = []
    for offset, bio_tag in zip(offset_mapping, bio_tags):
        if offset == (0, 0):
            label_ids.append(-100)
        else:
            label_ids.append(label2id[bio_tag])

    return {
        "text": example["text"],
        "tokens": tokens,
        "ner_tags": bio_tags,
        "offset_mapping": offset_mapping,
        "input_ids": encoding["input_ids"],
        "labels": label_ids,
        "attention_mask": encoding["attention_mask"],
        "token_type_ids": encoding["token_type_ids"]
    }

# ---- Preprocess for inference ----
def preprocess_for_inference(text, tokenizer):

    encoding = tokenizer(
        text=text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="longest",
        add_special_tokens=True
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    offset_mapping = encoding["offset_mapping"][0].tolist()

    return {
        "text": text,
        "tokens": tokens,
        "offset_mapping": offset_mapping,
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "token_type_ids": encoding["token_type_ids"]
    }
