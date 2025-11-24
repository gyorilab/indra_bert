from typing import List

from transformers import AutoTokenizer

from .shared import normalize_entity_tags


def preprocess_for_inference(text: str, tokenizer: AutoTokenizer):
    text = normalize_entity_tags(text)
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
        return_offsets_mapping=True,
    )


def preprocess_for_inference_batch(texts: List[str], tokenizer: AutoTokenizer, max_length: int = 512):
    texts_normalized = [normalize_entity_tags(t) for t in texts]
    return tokenizer(
        texts_normalized,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
