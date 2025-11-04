"""
PTM Site Extractor Module

This module provides functionality for extracting post-translational modification (PTM) sites
from biological text, specifically designed to work with INDRA statements.

The module includes:
- PTM site detection using token classification
- Support for various PTM types (phosphorylation, acetylation, methylation, etc.)
- Integration with existing INDRA statement extraction pipeline
"""

from .inference import PTMSiteExtractor
from .preprocess import preprocess_for_training
from .train import main as train_ptm_model

__all__ = ['PTMSiteExtractor', 'preprocess_for_training', 'train_ptm_model']
