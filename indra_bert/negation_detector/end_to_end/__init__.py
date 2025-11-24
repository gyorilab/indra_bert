"""End-to-end negation detection: predicts cues and scopes simultaneously using BIO tagging."""

from .model import NegationDetectorModel
from .inference import NegationDetector
from .preprocess import (
    load_negation_dataset,
    build_label_mappings,
    tokenize_negation_dataset,
)
from .postprocess import extract_spans_from_encoding, extend_outer_scopes
from .train import NegationTrainer, MultiHeadDataCollator

__all__ = [
    "NegationDetectorModel",
    "NegationDetector",
    "load_negation_dataset",
    "build_label_mappings",
    "tokenize_negation_dataset",
    "extract_spans_from_encoding",
    "extend_outer_scopes",
    "NegationTrainer",
    "MultiHeadDataCollator",
]

