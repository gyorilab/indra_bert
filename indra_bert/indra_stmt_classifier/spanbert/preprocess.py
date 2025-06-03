import json
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from transformers import AutoTokenizer
import re
from indra_bert.ner_agent_detector.model import AgentNERModel
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)


# ---- Helpers ----
def normalize_entity_tags(text: str) -> str:
    """
    Replace all opening <...> tags with <e> and all closing </...> tags with </e>
    """
    # Replace any opening tag like <subj>, <obj>, <xyz> with <e>
    text = re.sub(r"<[^/][^>]*>", "<e>", text)
    # Replace any closing tag like </subj>, </obj>, </xyz> with </e>
    text = re.sub(r"</[^>]+>", "</e>", text)
    return text

def mask_annotated_text(text, mask_token='[ENTITY]'):
    return re.sub(r"<e>.*?</e>", f"<e>{mask_token}</e>", text)

def extract_entity_char_spans(text: str) -> List[Dict[str, int]]:
    """Extract start and end positions for all <e>...</e> entities."""
    spans = []
    for match in re.finditer(r"<e>(.*?)</e>", text):
        spans.append({"text": match.group(1), "start": match.start(1), "end": match.end(1)})
    return spans

def spans_overlap(span1, span2):
    return not (span1['end'] < span2['start'] or span2['end'] < span1['start'])


# ---- Load + preprocess raw training data ----
def load_and_preprocess_raw_data(input_path):
    examples = []
    stmt2id = OrderedDict()
    ner_model = AgentNERModel("thomaslim6793/indra_bert_ner_agent_detection")

    stmt2id['No_Relation'] = len(stmt2id)  # Add a special label for no relation

    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Loading and preprocessing")):
            try:
                obj = json.loads(line)
                annotated_text = normalize_entity_tags(obj["annotated_text"])
                label = obj["statement"]["type"]

                if label not in stmt2id:
                    stmt2id[label] = len(stmt2id)

                ner_pred_annotated_text = ner_model.predict(obj['text'])['annotated_text']

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
    masked_texts = []

    for text in examples["text"]:
        masked_text = mask_annotated_text(text)
        masked_texts.append(masked_text)

    # Tokenize the masked texts
    encoding = tokenizer(
        masked_texts,
        truncation=True,
        max_length=512,
        padding=False,
        add_special_tokens=True,
        return_token_type_ids=True,
    )

    encoding["masked_text"] = masked_texts
    encoding["labels"] = examples["stmt_label_id"]
    encoding["stmt_label"] = examples["stmt_label"]
    
    return encoding
    
def preprocess_negative_examples_for_model(batch, stmt2id, tokenizer):
    negative_examples = []

    texts = batch["text"]
    ner_texts = batch["ner_pred_annotated_text"]

    for gold_text, ner_pred_text in zip(texts, ner_texts):
        gold_spans = extract_entity_char_spans(gold_text)
        ner_spans = extract_entity_char_spans(ner_pred_text)
        raw_text = re.sub(r"</?e>", "", gold_text)

        if len(gold_spans) != 2:
            continue  # Skip malformed samples

        gold_ent1, gold_ent2 = gold_spans

        for anchor_ent, blocked_ent in [(gold_ent1, gold_ent2), (gold_ent2, gold_ent1)]:
            for pred_span in ner_spans:
                # Skip if overlapping with either the anchor or the blocked gold
                if spans_overlap(pred_span, anchor_ent) or spans_overlap(pred_span, blocked_ent):
                    continue

                for ent1, ent2 in [(anchor_ent['text'], pred_span['text']), (pred_span['text'], anchor_ent['text'])]:
                    try:
                        temp_text = raw_text
                        temp_text = re.sub(rf"\b{re.escape(ent1)}\b", f"<e>{ent1}</e>", temp_text, count=1)
                        temp_text = re.sub(rf"\b{re.escape(ent2)}\b", f"<e>{ent2}</e>", temp_text, count=1)

                        if temp_text.count("<e>") != 2:
                            continue

                        masked_text = mask_annotated_text(temp_text)
                        enc = tokenizer(
                            masked_text,
                            truncation=True,
                            max_length=512,
                            padding=False,
                            add_special_tokens=True,
                            return_token_type_ids=True,
                        )

                        negative_examples.append({
                            "input_ids": enc["input_ids"],
                            "attention_mask": enc["attention_mask"],
                            "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
                            "labels": stmt2id["No_Relation"],
                            "stmt_label": "No_Relation",
                            "masked_text": masked_text,
                        })
                        break
                    except Exception as e:
                        logger.error(f"Error processing negative example: {e}")
                        continue

    return {key: [d[key] for d in negative_examples] for key in negative_examples[0]}

# ---- Tokenize for inference ----
def preprocess_for_inference(text, tokenizer):
    masked_text = mask_annotated_text(text)
    enc = tokenizer(
        masked_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="longest",
        add_special_tokens=True
    )
    enc["masked_text"] = masked_text
    return enc

def preprocess_for_inference_batch(texts: list[str], tokenizer, max_length=512):
    masked_texts = [mask_annotated_text(text) for text in texts]
    enc = tokenizer(
        masked_texts,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc["masked_text"] = masked_texts
    return enc
