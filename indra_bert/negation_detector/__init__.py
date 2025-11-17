"""Negation detector module for extracting negation cues and scopes.

Two implementations:
- end_to_end: Predicts cues and scopes simultaneously using BIO tagging
- two_step: First detects cues, then predicts scope for each cue independently
"""

from .end_to_end import NegationDetector as EndToEndNegationDetector
from .two_step import NegationDetectorModel as TwoStepNegationDetectorModel

__all__ = [
    "EndToEndNegationDetector",
    "TwoStepNegationDetectorModel",
]

