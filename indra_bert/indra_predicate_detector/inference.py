"""Inference utilities for the predicate detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .preprocess import ID2LABEL


@dataclass
class PredicateSpan:
    """Container for a predicted predicate span."""

    start: int  # character start index in untagged text (inclusive)
    end: int  # character end index in untagged text (exclusive)
    text: str
    score: Optional[float] = None
    tagged_start: Optional[int] = None
    tagged_end: Optional[int] = None


def strip_entity_tags_with_mapping(tagged_text: str) -> Tuple[str, List[int]]:
    """Remove ``<e>`` tags and return clean text plus index mapping."""
    clean_chars: List[str] = []
    mapping: List[int] = [0] * (len(tagged_text) + 1)
    i = 0
    clean_idx = 0

    while i < len(tagged_text):
        mapping[i] = clean_idx
        if tagged_text.startswith("<e>", i):
            i += 3
            continue
        if tagged_text.startswith("</e>", i):
            i += 4
            continue
        clean_chars.append(tagged_text[i])
        i += 1
        clean_idx += 1

    mapping[len(tagged_text)] = clean_idx
    return ''.join(clean_chars), mapping


class PredicateDetector:
    """Wrapper around the trained predicate span model."""

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            id2label=ID2LABEL,
        ).to(self.device)
        self.model.eval()

    def _decode(self, text_with_tags: str) -> List[PredicateSpan]:
        clean_text, mapping = strip_entity_tags_with_mapping(text_with_tags)

        encoding = self.tokenizer(
            text_with_tags,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            add_special_tokens=True,
        )
        offsets = encoding.pop("offset_mapping")[0].tolist()
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.inference_mode():
            outputs = self.model(**encoding)
            logits = outputs.logits[0]
            probs = logits.softmax(dim=-1)
            pred_ids = probs.argmax(dim=-1).cpu().tolist()

        spans: List[PredicateSpan] = []
        current_tagged_start: Optional[int] = None
        current_tagged_end: Optional[int] = None
        scores: List[float] = []

        for idx, (label_id, (start_char, end_char)) in enumerate(zip(pred_ids, offsets)):
            label = ID2LABEL[label_id]
            if start_char == end_char:  # special/padding token
                continue

            score = probs[idx, label_id].item()

            if label == "B-PRED":
                if current_tagged_start is not None and current_tagged_end is not None:
                    clean_start = mapping[current_tagged_start]
                    clean_end = mapping[current_tagged_end]
                    spans.append(
                        PredicateSpan(
                            start=clean_start,
                            end=clean_end,
                            text=clean_text[clean_start:clean_end],
                            score=float(sum(scores) / len(scores)),
                            tagged_start=current_tagged_start,
                            tagged_end=current_tagged_end,
                        )
                    )
                current_tagged_start = start_char
                current_tagged_end = end_char
                scores = [score]
            elif label == "I-PRED" and current_tagged_start is not None:
                current_tagged_end = end_char
                scores.append(score)
            else:
                if current_tagged_start is not None and current_tagged_end is not None:
                    clean_start = mapping[current_tagged_start]
                    clean_end = mapping[current_tagged_end]
                    spans.append(
                        PredicateSpan(
                            start=clean_start,
                            end=clean_end,
                            text=clean_text[clean_start:clean_end],
                            score=float(sum(scores) / len(scores)),
                            tagged_start=current_tagged_start,
                            tagged_end=current_tagged_end,
                        )
                    )
                current_tagged_start = None
                current_tagged_end = None
                scores = []

        if current_tagged_start is not None and current_tagged_end is not None:
            clean_start = mapping[current_tagged_start]
            clean_end = mapping[current_tagged_end]
            spans.append(
                PredicateSpan(
                    start=clean_start,
                    end=clean_end,
                    text=clean_text[clean_start:clean_end],
                    score=float(sum(scores) / len(scores)),
                    tagged_start=current_tagged_start,
                    tagged_end=current_tagged_end,
                )
            )

        return spans

    def predict(self, text: str) -> List[PredicateSpan]:
        """Return predicate spans for a single sentence."""
        return self._decode(text)

    def predict_batch(self, texts: List[str]) -> List[List[PredicateSpan]]:
        """Batch version of :meth:`predict` for convenience."""
        return [self._decode(text) for text in texts]
