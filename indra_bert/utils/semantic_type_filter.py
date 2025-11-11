from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Default hard-invalid entity type pairs (never allowed).
DEFAULT_FORBIDDEN_TYPE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("SequenceVariant", "SequenceVariant"),
    ("SequenceVariant", "ChemicalEntity"),
    ("SequenceVariant", "DiseaseOrPhenotypicFeature"),
    ("SequenceVariant", "CellLine"),
    ("SequenceVariant", "OrganismTaxon"),
    ("DiseaseOrPhenotypicFeature", "DiseaseOrPhenotypicFeature"),
)

# Default per-statement-label allowed type pairs.
DEFAULT_ALLOWED_TYPE_PAIRS_BY_LABEL: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "Phosphorylation": (("GeneOrGeneProduct", "GeneOrGeneProduct"),),
    "Dephosphorylation": (("GeneOrGeneProduct", "GeneOrGeneProduct"),),
    "Ubiquitination": (("GeneOrGeneProduct", "GeneOrGeneProduct"),),
    "Acetylation": (("GeneOrGeneProduct", "GeneOrGeneProduct"),),
    "Complex": (
        ("GeneOrGeneProduct", "GeneOrGeneProduct"),
        ("GeneOrGeneProduct", "ChemicalEntity"),
        ("ChemicalEntity", "ChemicalEntity"),
    ),
    "Activation": (
        ("GeneOrGeneProduct", "GeneOrGeneProduct"),
        ("ChemicalEntity", "GeneOrGeneProduct"),
    ),
    "Inhibition": (
        ("GeneOrGeneProduct", "GeneOrGeneProduct"),
        ("ChemicalEntity", "GeneOrGeneProduct"),
    ),
    "IncreaseAmount": (
        ("GeneOrGeneProduct", "GeneOrGeneProduct"),
        ("ChemicalEntity", "GeneOrGeneProduct"),
    ),
    "DecreaseAmount": (
        ("GeneOrGeneProduct", "GeneOrGeneProduct"),
        ("ChemicalEntity", "GeneOrGeneProduct"),
    ),
    "Translocation": (
        ("GeneOrGeneProduct", "GeneOrGeneProduct"),
        ("GeneOrGeneProduct", "DiseaseOrPhenotypicFeature"),
    ),
}

# Used when no label-specific schema exists.
DEFAULT_FALLBACK_ALLOWED_TYPE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("GeneOrGeneProduct", "GeneOrGeneProduct"),
    ("ChemicalEntity", "GeneOrGeneProduct"),
    ("ChemicalEntity", "ChemicalEntity"),
)


def _normalize_type_pair(pair: Sequence[str]) -> Tuple[str, str]:
    if len(pair) != 2:
        raise ValueError("Entity type pairs must have exactly two entries")
    a, b = pair
    # Order-invariant: store sorted
    return tuple(sorted((a, b)))  # type: ignore[arg-type]


@dataclass
class TypeConstraintConfig:
    """
    Configuration for semantic / type-based post-filtering of statements.
    """

    # Additional globally forbidden type pairs (beyond defaults).
    extra_forbidden_type_pairs: Sequence[Sequence[str]] = field(default_factory=list)
    use_default_forbidden_pairs: bool = True

    # If True, drop statements where either argument is missing a type.
    require_entity_types: bool = True

    # If True, drop statements where both arguments point to the same span.
    drop_duplicate_span_pairs: bool = True

    # Additional allowed pairs for specific statement labels.
    # Keys should match stmt_pred["label"].
    extra_allowed_type_pairs_by_label: Dict[str, Tuple[Tuple[str, str], ...]] = field(default_factory=dict)
    use_default_allowed_pairs_by_label: bool = True

    # Fallback allowed pairs when no label-specific schema exists.
    fallback_allowed_type_pairs: Tuple[Tuple[str, str], ...] = DEFAULT_FALLBACK_ALLOWED_TYPE_PAIRS


@dataclass
class TypeFilterResult:
    kept_statements: List[Dict[str, Any]]
    drop_reasons: Counter


def filter_statements_by_type_constraints(
    statements: Iterable[Dict[str, Any]],
    config: Optional[TypeConstraintConfig] = None,
) -> TypeFilterResult:
    """
    Filter structured statements using entity-type compatibility constraints.

    Expects each statement to contain:
      - "entity_pair": [ent1, ent2]
      - each ent has "type", and optionally "start"/"end"
      - "stmt_pred": {"label": ...}  (used for label-specific schemas)
    """

    cfg = config or TypeConstraintConfig()

    # Build forbidden set
    forbidden_pairs = set()
    if cfg.use_default_forbidden_pairs:
        forbidden_pairs.update(_normalize_type_pair(p) for p in DEFAULT_FORBIDDEN_TYPE_PAIRS)
    forbidden_pairs.update(_normalize_type_pair(p) for p in cfg.extra_forbidden_type_pairs)

    # Build allowed-by-label map
    allowed_by_label: Dict[str, set[Tuple[str, str]]] = {}
    if cfg.use_default_allowed_pairs_by_label:
        for label, pairs in DEFAULT_ALLOWED_TYPE_PAIRS_BY_LABEL.items():
            allowed_by_label.setdefault(label, set()).update(
                _normalize_type_pair(p) for p in pairs
            )
    for label, pairs in cfg.extra_allowed_type_pairs_by_label.items():
        allowed_by_label.setdefault(label, set()).update(
            _normalize_type_pair(p) for p in pairs
        )

    # Fallback allowed pairs
    fallback_allowed = {_normalize_type_pair(p) for p in cfg.fallback_allowed_type_pairs}

    kept: List[Dict[str, Any]] = []
    drops: Counter = Counter()

    for stmt in statements:
        pair = stmt.get("entity_pair")
        if not pair or len(pair) != 2:
            drops["invalid_entity_pair"] += 1
            continue

        ent1, ent2 = list(pair)

        # Type presence
        if cfg.require_entity_types and (not ent1.get("type") or not ent2.get("type")):
            drops["missing_entity_type"] += 1
            continue

        # Same span → likely duplication / coref noise
        if cfg.drop_duplicate_span_pairs:
            if ent1.get("start") == ent2.get("start") and ent1.get("end") == ent2.get("end"):
                drops["duplicate_entity_span"] += 1
                continue

        type_pair = _normalize_type_pair((ent1.get("type", ""), ent2.get("type", "")))

        # Globally forbidden combinations
        if type_pair in forbidden_pairs:
            drops["forbidden_type_pair"] += 1
            continue

        # Label-specific schema
        label = str(stmt.get("stmt_pred", {}).get("label", "")).strip()
        allowed_for_label = allowed_by_label.get(label)

        if allowed_for_label:
            if type_pair not in allowed_for_label:
                drops[f"disallowed_for_label:{label}"] += 1
                continue
        else:
            # Use fallback schema when no label-specific constraints defined
            if fallback_allowed and type_pair not in fallback_allowed:
                drops["disallowed_fallback_pair"] += 1
                continue

        # Require at least one gene-like entity, unless it's chem-chem
        if (
            ent1.get("type") != "GeneOrGeneProduct"
            and ent2.get("type") != "GeneOrGeneProduct"
            and type_pair != _normalize_type_pair(("ChemicalEntity", "ChemicalEntity"))
        ):
            drops["no_gene_or_geneproduct"] += 1
            continue

        kept.append(stmt)

    return TypeFilterResult(kept_statements=kept, drop_reasons=drops)
