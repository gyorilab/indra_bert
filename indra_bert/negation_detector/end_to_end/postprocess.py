"""
Postprocessing utilities for extracting cue and scope spans from predictions.
"""
from typing import List, Dict, Any, Tuple


def extend_outer_scopes(
    scopes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Post-process scopes to extend outer scopes when they contain inner scopes.
    
    For nested negations like "it is not true that A does not activate B",
    the outer scope should extend to include the entire embedded clause.
    
    Args:
        scopes: List of scope dictionaries with 'start' and 'end' keys
    
    Returns:
        List of scopes with outer scopes extended
    """
    if len(scopes) < 2:
        return scopes
    
    # Sort by start position, then by end position (descending for outer scopes)
    sorted_scopes = sorted(scopes, key=lambda s: (s['start'], -s['end']))
    extended_scopes = []
    
    for i, scope in enumerate(sorted_scopes):
        # Check if this scope is nested within any previous scope
        is_nested = False
        for prev_scope in extended_scopes:
            if prev_scope['start'] <= scope['start'] and scope['end'] <= prev_scope['end']:
                # This scope is nested within prev_scope
                # Extend prev_scope to ensure it covers the full embedded clause
                # The outer scope should extend to at least the end of the inner scope
                if prev_scope['end'] < scope['end']:
                    prev_scope['end'] = scope['end']
                    # Update text if available
                    if 'text' in prev_scope and 'text' in scope:
                        # Reconstruct text from original (we'd need the original text for this)
                        pass
                is_nested = True
                break
        
        if not is_nested:
            extended_scopes.append(scope.copy())
        else:
            # Still add the inner scope
            extended_scopes.append(scope.copy())
    
    return extended_scopes


def extract_spans_from_encoding(
    tokens: List[str],
    offset_mapping: List[tuple],
    predicted_label_ids: List[int],
    id2label: Dict[int, str],
    text: str,
    span_type: str = "cue",  # "cue" or "scope"
) -> List[Dict[str, Any]]:
    """
    Extract spans from token predictions.
    
    Args:
        tokens: List of token strings
        offset_mapping: List of (start, end) tuples for each token
        predicted_label_ids: List of predicted label IDs
        id2label: Mapping from label IDs to label strings
        text: Original text
        span_type: "cue" or "scope" to filter spans
    
    Returns:
        List of span dictionaries with 'start', 'end', 'text', 'type'
    """
    spans = []
    current_span = None
    
    for token, offset, label_id in zip(tokens, offset_mapping, predicted_label_ids):
        if label_id == -100:
            continue
        
        start_char, end_char = offset
        
        if start_char is None or end_char is None or (start_char == end_char):
            if current_span:
                spans.append(current_span)
                current_span = None
            continue
        
        label = id2label.get(label_id, "O")
        
        # Check if this label matches the span_type we're looking for
        if label == "O" or not label.endswith(f"-{span_type}"):
            if current_span:
                spans.append(current_span)
                current_span = None
            continue
        
        # Extract tag (B or I)
        tag = label.split("-")[0]
        
        if tag == "B":
            if current_span:
                spans.append(current_span)
            current_span = {
                "start": start_char,
                "end": end_char,
                "text": text[start_char:end_char],
                "type": span_type,
            }
        elif tag == "I":
            if current_span:
                current_span["end"] = end_char
                current_span["text"] = text[current_span["start"]:end_char]
            else:
                # I without B → treat as a new span
                current_span = {
                    "start": start_char,
                    "end": end_char,
                    "text": text[start_char:end_char],
                    "type": span_type,
                }
    
    if current_span:
        spans.append(current_span)
    
    return spans

