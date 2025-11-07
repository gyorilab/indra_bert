"""Pre-processing utilities for predicate detector training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Any

from tqdm import tqdm

PRED_START = "<predicate>"
PRED_END = "</predicate>"
LABEL2ID = {"O": 0, "B-PRED": 1, "I-PRED": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class RawPredicateExample:
    """Minimal representation of an annotated sentence."""

    id: int
    text: str  # entity tags preserved, predicate tags removed (if present)
    predicate_span: Tuple[int, int] | None
    matches_hash: str
    source_hash: str


def parse_tagged_text(tagged_text: str) -> Tuple[str, Tuple[int, int] | None]:
    """Return text without predicate tags and the predicate char span (if any).

    If no ``<predicate>`` markers are present, the second element of the tuple is
    ``None`` and the text is returned unchanged.
    """
    if PRED_START not in tagged_text:
        return tagged_text, None

    clean_chars: List[str] = []
    i = 0
    pred_start = None
    pred_end = None

    while i < len(tagged_text):
        if tagged_text.startswith(PRED_START, i):
            if pred_start is not None:
                raise ValueError("Multiple <predicate> tags found in example")
            i += len(PRED_START)
            pred_start = len(clean_chars)
            continue
        if tagged_text.startswith(PRED_END, i):
            if pred_start is None:
                raise ValueError("Closing </predicate> without opening tag")
            i += len(PRED_END)
            pred_end = len(clean_chars)
            continue

        clean_chars.append(tagged_text[i])
        i += 1

    if pred_start is None or pred_end is None:
        raise ValueError("Malformed predicate tags in example")

    clean_text = ''.join(clean_chars)
    return clean_text, (pred_start, pred_end)


def read_annotated_jsonl(path: Path) -> Iterable[dict]:
    """Yield entries from the filtered JSONL file."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def json_to_raw_example(entry: dict, example_id: int) -> RawPredicateExample:
    """Convert a JSON entry into :class:`RawPredicateExample`."""
    output_text = entry["output_text"]
    text, span = parse_tagged_text(output_text)
    return RawPredicateExample(
        id=example_id,
        text=text,
        predicate_span=span,
        matches_hash=str(entry.get("matches_hash", "")),
        source_hash=str(entry.get("source_hash", "")),
    )


def load_and_preprocess_raw_data(input_path: Path) -> List[RawPredicateExample]:
    """Load the filtered JSONL and produce raw predicate examples."""
    examples: List[RawPredicateExample] = []

    for idx, entry in enumerate(tqdm(read_annotated_jsonl(input_path), desc="Loading predicate annotations")):
        try:
            examples.append(json_to_raw_example(entry, idx))
        except Exception as exc:  # pylint: disable=broad-except
            tqdm.write(f"Skipping example {idx} due to parse error: {exc}")
            continue

    return examples


def _labels_from_offsets(
    offsets: Sequence[Tuple[int, int]],
    predicate_span: Tuple[int, int] | None,
) -> List[int]:
    """Generate label IDs aligned to tokenizer offsets."""
    if predicate_span is None:
        return [LABEL2ID["O"] for _ in offsets]

    pred_start, pred_end = predicate_span
    labels: List[int] = []
    started = False

    for start_char, end_char in offsets:
        if start_char == end_char:
            labels.append(LABEL2ID["O"])
            continue
        if start_char >= pred_end or end_char <= pred_start:
            labels.append(LABEL2ID["O"])
            continue
        labels.append(LABEL2ID["B-PRED"] if not started else LABEL2ID["I-PRED"])
        started = True

    return labels


def preprocess_examples_for_model(
    examples: Sequence[RawPredicateExample],
    tokenizer,
    max_length: int = 256,
    padding: bool | str = "longest",
    truncation: bool = True,
) -> Dict[str, Any]:
    """Tokenise predicate examples and produce BIO label IDs.

    Parameters
    ----------
    examples:
        Iterable of :class:`RawPredicateExample` produced by
        :func:`load_and_preprocess_raw_data`.
    tokenizer:
        Hugging Face tokenizer used by the downstream model.
    max_length:
        Maximum sequence length passed to the tokenizer.
    padding:
        Padding strategy (``True``/``False``/``"longest"``) forwarded to the tokenizer.
    truncation:
        Whether to truncate sequences longer than ``max_length``.

    Returns
    -------
    Dict[str, Any]
        Dictionary compatible with Hugging Face ``Dataset`` expectations, containing
        ``input_ids``, ``attention_mask``, ``labels`` and provenance metadata.
    """
    texts = [ex.text for ex in examples]
    encoding = tokenizer(
        texts,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )

    offsets_list = encoding["offset_mapping"]

    labels: List[List[int]] = []
    processed_offsets: List[List[Tuple[int, int]]] = []
    for offsets, ex in zip(offsets_list, examples):
        labels.append(_labels_from_offsets(offsets, ex.predicate_span))
        processed_offsets.append([(int(start), int(end)) for start, end in offsets])

    encoding.pop("offset_mapping")
    encoding["labels"] = labels
    encoding["matches_hash"] = [ex.matches_hash for ex in examples]
    encoding["source_hash"] = [ex.source_hash for ex in examples]
    encoding["example_id"] = [ex.id for ex in examples]
    encoding["offset_mapping"] = processed_offsets
    encoding["text"] = [ex.text for ex in examples]
    encoding["tokens"] = [tokenizer.convert_ids_to_tokens(ids) for ids in encoding["input_ids"]]

    return encoding
