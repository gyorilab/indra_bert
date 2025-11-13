from typing import List, Dict, Any, Tuple


def extract_trigger_spans_from_encoding(
    tokens: List[str],
    offset_mapping: List[Tuple[int, int]],
    predicted_label_ids: List[int],
    id2label: Dict[int, str],
    text: str
) -> List[Dict[str, Any]]:
    """
    Extract trigger spans from token predictions.
    
    Returns list of trigger spans with their roles.
    """
    spans = []
    current_span = None
    current_role = None
    
    for i, (label_id, (tok_start, tok_end)) in enumerate(zip(predicted_label_ids, offset_mapping)):
        if tok_start is None or tok_end is None:
            continue
        
        label = id2label.get(label_id, "O")
        
        if label.startswith("B-trigger-"):
            # Start of new trigger span
            if current_span is not None:
                # Save previous span
                spans.append(current_span)
            
            # Extract role from label (B-trigger-Head or B-trigger-Tail)
            role = label.replace("B-trigger-", "")
            current_role = role
            current_span = {
                "start": tok_start,
                "end": tok_end,
                "text": text[tok_start:tok_end] if tok_start < len(text) and tok_end <= len(text) else "",
                "role": role,
            }
        elif label.startswith("I-trigger-"):
            # Continuation of trigger span
            if current_span is not None:
                # Extend the span
                current_span["end"] = tok_end
                if tok_start < len(text) and tok_end <= len(text):
                    current_span["text"] = text[current_span["start"]:tok_end]
        else:
            # O or other - end current span
            if current_span is not None:
                spans.append(current_span)
                current_span = None
                current_role = None
    
    # Don't forget the last span
    if current_span is not None:
        spans.append(current_span)
    
    return spans

