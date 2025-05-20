import json
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from transformers import AutoTokenizer
import re

# ---- Normalize annotation tags ----
def normalize_entity_tags(text: str) -> str:
    """
    Replace all opening <...> tags with <e> and all closing </...> tags with </e>
    """
    # Replace any opening tag like <subj>, <obj>, <xyz> with <e>
    text = re.sub(r"<[^/][^>]*>", "<e>", text)
    # Replace any closing tag like </subj>, </obj>, </xyz> with </e>
    text = re.sub(r"</[^>]+>", "</e>", text)
    return text

# ---- Load + preprocess raw training data ----
def load_and_preprocess_raw_data(input_path):
    examples = []
    stmt2id = OrderedDict()

    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Loading and preprocessing")):
            try:
                obj = json.loads(line)
                annotated_text = normalize_entity_tags(obj["annotated_text"])
                label = obj["statement"]["type"]

                if label not in stmt2id:
                    stmt2id[label] = len(stmt2id)

                examples.append({
                    "id": idx,
                    "text": annotated_text,
                    "stmt_label": label,
                    "stmt_label_id": stmt2id[label]
                })
            except Exception as e:
                print(f"Error processing line {idx}: {e}")
                continue

    return examples, stmt2id

# ---- Tokenize training examples ----
def preprocess_examples_for_model(examples, tokenizer):
    encoding = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="longest",
        add_special_tokens=True
    )
    encoding["labels"] = examples["stmt_label_id"]
    encoding["stmt_label"] = examples["stmt_label"]  # Optional for inspection
    return encoding

# ---- Tokenize for inference ----
def preprocess_for_inference(text, tokenizer):
    # No preprocess needed currently, but keeping for consistency
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="longest",
        add_special_tokens=True
    )

def preprocess_for_inference_batch(texts: list[str], tokenizer, max_length=512):
    """
    Tokenize a batch of input texts for classification.
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return enc
