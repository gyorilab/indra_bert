"""Post-processing utilities for predicate detector outputs."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .preprocess import ID2LABEL, LABEL2ID


def decode_bio_labels(tokens: Sequence[str], labels: Sequence[int]) -> List[dict]:
    """Convert BIO label ids back into token-index spans."""
    spans: List[dict] = []
    current_start = None

    for idx, label_id in enumerate(labels):
        label_name = ID2LABEL.get(label_id, "O")
        if label_name == "B-PRED":
            if current_start is not None:
                spans.append({"start": current_start, "end": idx, "label": "PRED"})
            current_start = idx
        elif label_name == "I-PRED":
            if current_start is None:
                current_start = idx
        else:
            if current_start is not None:
                spans.append({"start": current_start, "end": idx, "label": "PRED"})
                current_start = None

    if current_start is not None:
        spans.append({"start": current_start, "end": len(labels), "label": "PRED"})

    return spans


def labels_to_bio_sequence(labels: Iterable[str]) -> List[int]:
    """Convert BIO label names to label ids."""
    return [LABEL2ID[label] for label in labels]


def spans_from_offsets(
    offsets: Sequence[Tuple[int, int]],
    label_ids: Sequence[int],
    text: str,
) -> List[Tuple[int, int, str]]:
    """Extract character-level spans from BIO label ids and offsets."""
    spans: List[Tuple[int, int, str]] = []
    current_start = None
    current_end = None

    for (start_char, end_char), label_id in zip(offsets, label_ids):
        label_name = ID2LABEL.get(label_id, "O")
        if start_char == end_char:
            continue
        if label_name == "B-PRED":
            if current_start is not None and current_end is not None:
                spans.append((current_start, current_end, text[current_start:current_end]))
            current_start = start_char
            current_end = end_char
        elif label_name == "I-PRED" and current_start is not None:
            current_end = end_char
        else:
            if current_start is not None and current_end is not None:
                spans.append((current_start, current_end, text[current_start:current_end]))
            current_start = None
            current_end = None

    if current_start is not None and current_end is not None:
        spans.append((current_start, current_end, text[current_start:current_end]))

    return spans
