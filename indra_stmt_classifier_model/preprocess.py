import json
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from transformers import AutoTokenizer

# ---- Load + preprocess raw training data ----
def load_and_preprocess_raw_data(input_path,):
    examples = []
    stmt2id = OrderedDict()

    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Loading and preprocessing")):
            try:
                obj = json.loads(line)
                raw_text = obj["text"]
                label = obj["statement"]["type"]

                if label not in stmt2id:
                    stmt2id[label] = len(stmt2id)

                full_text = raw_text

                examples.append({
                    "id": idx,
                    "text": full_text,
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

    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
        add_special_tokens=True
    )
