import json
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from transformers import AutoTokenizer
import re
from indra_bert.ner_agent_detector.model import AgentNERModel
import logging

logger = logging.getLogger(__name__)


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
                    "annotated_text": annotated_text,
                    "stmt_label": label,
                    "stmt_label_id": stmt2id[label],
                    "ner_pred_annotated_text": ner_pred_annotated_text
                })
            except Exception as e:
                print(f"Error processing line {idx}: {e}")
                continue

    return examples, stmt2id

def map_char_span_to_token_span(char_span, offset_mapping):
    start_char, end_char = char_span
    token_start = token_end = None
    for idx, (token_start_char, token_end_char) in enumerate(offset_mapping):
        if token_start is None and token_start_char <= start_char < token_end_char:
            token_start = idx
        if token_start is not None and token_start_char < end_char <= token_end_char:
            token_end = idx + 1
            break

    if token_start is None or token_end is None:
        return None  
    
    return [token_start, token_end]

# ---- Tokenize training examples ----
def preprocess_examples_for_model(examples, tokenizer):
    encoding = tokenizer(
        examples["annotated_text"],
        truncation=True,
        max_length=512,
        padding=False,
        add_special_tokens=True,
        return_token_type_ids=True,
        return_offsets_mapping=True
    )
    encoding["labels"] = examples["stmt_label_id"]
    encoding["stmt_label"] = examples["stmt_label"]  # Optional

    # Extract character-level entity spans for each example
    entity_char_spans_batch = [extract_entity_spans(text) for text in examples["annotated_text"]]
    entity_char_spans_batch = [
        [[span["start"], span["end"]] for span in spans]
        for spans in entity_char_spans_batch
    ]
    encoding["entity_char_spans"] = entity_char_spans_batch

    # Now convert char spans to token spans using offset_mapping per example
    token_spans_batch = []
    for i, spans in enumerate(entity_char_spans_batch):
        token_spans = []
        offset_mapping = encoding["offset_mapping"][i]
        for span in spans:
            mapped = map_char_span_to_token_span(span, offset_mapping)
            if mapped is not None:
                token_spans.append(mapped)
        token_spans_batch.append(token_spans)

    encoding["entity_token_spans"] = token_spans_batch

    return encoding

import re
from typing import List, Dict

def extract_entity_spans(text: str) -> List[Dict[str, int]]:
    """Extract start and end positions for all <e>...</e> entities."""
    spans = []
    for match in re.finditer(r"<e>(.*?)</e>", text):
        spans.append({"text": match.group(1), "start": match.start(1), "end": match.end(1)})
    return spans

def spans_overlap(span1, span2):
    return not (span1['end'] <= span2['start'] or span2['end'] <= span1['start'])

def preprocess_negative_examples_for_model(batch, stmt2id, tokenizer):
    negative_examples = []

    texts = batch["annotated_text"]
    ner_texts = batch["ner_pred_annotated_text"]

    for gold_text, ner_pred_text in zip(texts, ner_texts):
        gold_spans = extract_entity_spans(gold_text)
        ner_spans = extract_entity_spans(ner_pred_text)
        raw_text = re.sub(r"</?e>", "", gold_text)

        gold_span_texts = {span["text"] for span in gold_spans}

        for i in range(len(ner_spans)):
            for j in range(i + 1, len(ner_spans)):
                span1, span2 = ner_spans[i], ner_spans[j]
                text1, text2 = span1["text"], span2["text"]

                # Ensure exactly one span matches gold by text (not just overlap)
                in_gold_1 = text1 in gold_span_texts
                in_gold_2 = text2 in gold_span_texts

                if in_gold_1 ^ in_gold_2:  # XOR logic: exactly one is gold

                    # Skip if distractor has any text overlap with other golds
                    distractor = text1 if not in_gold_1 else text2
                    if any(distractor in g or g in distractor for g in gold_span_texts):
                        continue

                    for ent1, ent2 in [(text1, text2), (text2, text1)]:
                        try:
                            temp_text = raw_text
                            temp_text = re.sub(rf"\b{re.escape(ent1)}\b", f"<e>{ent1}</e>", temp_text, count=1)
                            temp_text = re.sub(rf"\b{re.escape(ent2)}\b", f"<e>{ent2}</e>", temp_text, count=1)

                            if temp_text.count("<e>") != 2:
                                continue

                            enc = tokenizer(
                                temp_text,
                                truncation=True,
                                max_length=512,
                                padding=False,
                                add_special_tokens=True,
                                return_token_type_ids=True,
                                return_offsets_mapping=True
                            )

                            new_spans = extract_entity_spans(temp_text)
                            if len(new_spans) != 2:
                                continue

                            entity_char_spans = [[s["start"], s["end"]] for s in new_spans]
                            token_spans = [map_char_span_to_token_span(span, enc["offset_mapping"]) for span in entity_char_spans]

                            if None in token_spans:
                                logger.error(f"Invalid token spans for text: {temp_text}",
                                             f" entity_char_spans: {entity_char_spans}, offset_mapping: {enc['offset_mapping']}")
                                continue

                            negative_examples.append({
                                "input_ids": enc["input_ids"],
                                "attention_mask": enc["attention_mask"],
                                "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
                                "labels": stmt2id["No_Relation"],
                                "stmt_label": "No_Relation",
                                "entity_char_spans": entity_char_spans,
                                "entity_token_spans": token_spans
                            })
                            break  # accept only the first valid form
                        except Exception as e:
                            logger.error(f"Error processing hard negative: {e}")
                            continue

    if not negative_examples:
        return {"input_ids": [], "attention_mask": [], "labels": [], "stmt_label": []}

    return {key: [d[key] for d in negative_examples] for key in negative_examples[0]}

# ---- Tokenize for inference ----
def preprocess_for_inference(text, tokenizer):
    # No preprocess needed currently, but keeping for consistency
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="longest",
        add_special_tokens=True,
        return_offsets_mapping=True
    )
    entity_char_spans = extract_entity_spans(text)
    entity_char_spans = [
        [span["start"], span["end"]] for span in entity_char_spans
    ]
    enc["entity_char_spans"] = entity_char_spans
    token_spans = [
        [map_char_span_to_token_span(span, enc["offset_mapping"][0]) for span in entity_char_spans]
    ]
    enc["entity_token_spans"] = token_spans
    return enc

def preprocess_for_inference_batch(texts: list[str], tokenizer, max_length=512):
    """
    Tokenize a batch of input texts for classification.
    """
    enc = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    entity_char_spans_batch = [extract_entity_spans(text) for text in texts]
    entity_char_spans_batch = [
        [[span["start"], span["end"]] for span in spans]
        for spans in entity_char_spans_batch
    ]
    enc["entity_char_spans"] = entity_char_spans_batch

    token_spans_batch = []
    for i, spans in enumerate(entity_char_spans_batch):
        token_spans = []
        offset_mapping = enc["offset_mapping"][i]
        for span in spans:
            mapped = map_char_span_to_token_span(span, offset_mapping)
            if mapped is not None:
                token_spans.append(mapped)
        token_spans_batch.append(token_spans)

    enc["entity_token_spans"] = token_spans_batch
    return enc
