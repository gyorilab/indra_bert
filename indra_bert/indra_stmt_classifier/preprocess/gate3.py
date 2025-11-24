import json
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import Dataset
from transformers import AutoTokenizer

from .shared import (
    normalize_entity_tags,
    IGNORE_INDEX,
)


def load_indra_benchmark_dataset(path: Path) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            obj = json.loads(line)
            annotated = obj.get("annotated_text") or obj.get("text")
            annotated = normalize_entity_tags(annotated)
            examples.append(
                {
                    "id": idx,
                    "text": obj["text"],
                    "annotated_text": annotated,
                    "indra_label": obj.get("statement", {}).get("type", "unknown"),
                }
            )
    return examples


def build_indra_label_mapping(examples: Iterable[Dict[str, object]]) -> Dict[str, int]:
    labels = sorted({ex["indra_label"] for ex in examples})
    return {label: idx for idx, label in enumerate(labels)}


def tokenize_indra_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    indra2id: Dict[str, int],
) -> Dataset:
    def _process(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        enc = tokenizer(batch["annotated_text"], truncation=True, padding=False, max_length=512)
        gate1_labels = [IGNORE_INDEX] * len(batch["annotated_text"])
        gate2_labels = [IGNORE_INDEX] * len(batch["annotated_text"])
        gate3_labels = [indra2id[label] for label in batch["indra_label"]]

        enc["gate1_labels"] = gate1_labels
        enc["gate2_labels"] = gate2_labels
        enc["gate3_labels"] = gate3_labels
        enc["indra_label"] = batch["indra_label"]
        return enc

    return dataset.map(
        _process,
        batched=True,
        load_from_cache_file=False,
        desc="Tokenizing INDRA dataset for Gate3",
    )
