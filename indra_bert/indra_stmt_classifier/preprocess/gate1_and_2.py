import json
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from .shared import (
    normalize_entity_tags,
    BINARY_LABEL_MAPPING,
    IGNORE_INDEX,
)


def load_relation_binary_dataset(path: Path) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            obj = json.loads(line)
            annotated = obj.get("text_with_entity_tags") or obj.get("annotated_text") or obj.get("text")
            annotated = normalize_entity_tags(annotated)
            examples.append(
                {
                    "id": idx,
                    "text": obj["text"],
                    "annotated_text": annotated,
                    "label": int(obj["label"]),
                    "relation_subtype": obj.get("relation_subtype", "unknown"),
                    "source": obj.get("source"),
                    "doc_id": obj.get("doc_id"),
                    "sentence_idx": obj.get("sentence_idx"),
                }
            )
    return examples


def build_relation_subtype_mapping(examples: Iterable[Dict[str, object]]) -> Dict[str, int]:
    labels = sorted({ex["relation_subtype"] for ex in examples})
    return {label: idx for idx, label in enumerate(labels)}


def create_dataset_splits(examples: List[Dict[str, object]], seed: int = 42) -> DatasetDict:
    dataset = Dataset.from_list(examples)
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    val_test = split["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(train=split["train"], validation=val_test["train"], test=val_test["test"])


def tokenize_relation_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    subtype2id: Dict[str, int],
) -> Dataset:
    def _process(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        enc = tokenizer(batch["annotated_text"], truncation=True, padding=False, max_length=512)
        gate1_labels = [
            BINARY_LABEL_MAPPING["has_relation"] if lbl == 1 else BINARY_LABEL_MAPPING["no_relation"]
            for lbl in batch["label"]
        ]
        gate2_labels = [subtype2id[label] for label in batch["relation_subtype"]]
        gate3_labels = [IGNORE_INDEX] * len(gate1_labels)

        enc["gate1_labels"] = gate1_labels
        enc["gate2_labels"] = gate2_labels
        enc["gate3_labels"] = gate3_labels
        enc["relation_subtype"] = batch["relation_subtype"]
        enc["label"] = batch["label"]
        return enc

    return dataset.map(
        _process,
        batched=True,
        load_from_cache_file=False,
        desc="Tokenizing relation dataset for Gate1/2",
    )
