
def extract_spans_from_encoding(tokens, offset_mapping, predicted_label_ids, id2label, text):
    spans = []
    current_span = None
    current_type = None

    for token, offset, label_id in zip(tokens, offset_mapping, predicted_label_ids):
        if label_id == -100:
            continue    

        start_char, end_char = offset

        if start_char is None or end_char is None or (start_char == end_char):
            if current_span:
                spans.append(current_span)
                current_span = None
                current_type = None
            continue

        label = id2label[label_id]
        
        if label == "O":
            if current_span:
                spans.append(current_span)
                current_span = None
                current_type = None
            continue

        parts = label.split("-", maxsplit=1)
        tag = parts[0]
        entity_type = parts[1] if len(parts) > 1 else "entity"

        if tag == "B":
            if current_span:
                spans.append(current_span)
            current_span = {
                "start": start_char,
                "end": end_char,
                "text": text[start_char:end_char],
                "type": entity_type,
                "raw_type": entity_type.replace("_", " "),
            }
            current_type = entity_type

        elif tag == "I":
            if current_span and current_type == entity_type:
                current_span["end"] = end_char
                current_span["text"] = text[current_span["start"]:end_char]
            else:
                # I without B or mismatched type → treat as a new span
                current_span = {
                    "start": start_char,
                    "end": end_char,
                    "text": text[start_char:end_char],
                    "type": entity_type,
                    "raw_type": entity_type.replace("_", " "),
                }
                current_type = entity_type

    if current_span:
        spans.append(current_span)

    return spans
