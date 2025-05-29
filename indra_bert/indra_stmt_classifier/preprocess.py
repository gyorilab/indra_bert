import json
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from transformers import AutoTokenizer
import re
from indra_bert.ner_agent_detector.model import AgentNERModel

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
    ner_model = AgentNERModel("thomaslim6793/indra_bert_ner_agent_detection")

    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Loading and preprocessing")):
            try:
                obj = json.loads(line)
                annotated_text = normalize_entity_tags(obj["annotated_text"])
                label = obj["statement"]["type"]

                if label not in stmt2id:
                    stmt2id[label] = len(stmt2id)

                ner_pred_annotated_text = ner_model.predict(obj['text'])['annotated_text']
                stmt2id['No_Relation'] = len(stmt2id)  # Add a special label for no relation

                examples.append({
                    "id": idx,
                    "text": annotated_text,
                    "stmt_label": label,
                    "stmt_label_id": stmt2id[label],
                    "ner_pred_annotated_text": ner_pred_annotated_text
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
        padding=False,
        add_special_tokens=True,
        return_token_type_ids=True
    )
    encoding["labels"] = examples["stmt_label_id"]
    encoding["stmt_label"] = examples["stmt_label"]  # Optional for inspection
    return encoding

def preprocess_negative_examples_for_model(batch, stmt2id, tokenizer):
    negative_examples = []

    texts = batch["text"]
    ner_texts = batch["ner_pred_annotated_text"]

    for annotated_text, ner_pred_annotated_text in zip(texts, ner_texts):
        if not isinstance(annotated_text, str) or not isinstance(ner_pred_annotated_text, str):
            continue

        # Extract gold entity spans
        true_spans = re.findall(r"<e>(.*?)</e>", annotated_text)
        if len(true_spans) != 2:
            continue

        true_pair = set(true_spans)

        # Extract all predicted NER spans
        ner_spans = re.findall(r"<e>(.*?)</e>", ner_pred_annotated_text)

        for i in range(len(ner_spans)):
            for j in range(i + 1, len(ner_spans)):
                cand_pair = (ner_spans[i], ner_spans[j])
                if set(cand_pair) == true_pair:
                    continue

                # Remove all <e> tags
                temp_text = re.sub(r"<e>(.*?)</e>", r"\1", annotated_text)

                ent1, ent2 = cand_pair
                inserted = False

                for order in [(ent1, ent2), (ent2, ent1)]:
                    try:
                        temp_version = temp_text
                        temp_version = re.sub(
                            rf"\b{re.escape(order[0])}\b", f"<e>{order[0]}</e>", temp_version, count=1)
                        temp_version = re.sub(
                            rf"\b{re.escape(order[1])}\b", f"<e>{order[1]}</e>", temp_version, count=1)

                        enc = tokenizer(
                            temp_version,
                            truncation=True,
                            max_length=512,
                            padding=False,
                            add_special_tokens=True,
                            return_token_type_ids=True
                        )

                        negative_examples.append({
                            "input_ids": enc["input_ids"],
                            "attention_mask": enc["attention_mask"],
                            "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
                            "labels": stmt2id["No_Relation"],
                            "stmt_label": "No_Relation"
                        })

                        inserted = True
                        break  # ✅ stop trying once one order worked
                    except Exception:
                        continue

                # Optionally, log failed insertion attempts here if inserted is still False

    if not negative_examples:
        return {"input_ids": [], "attention_mask": [], "labels": [], "stmt_label": []}

    return {key: [d[key] for d in negative_examples] for key in negative_examples[0]}

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
