
def extract_spans_from_encoding(tokens, offset_mapping, predicted_label_ids, id2label, text):
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

        label = id2label[label_id]
        
        if label == "O":
            if current_span:
                spans.append(current_span)
                current_span = None
            continue

        tag = label.split("-", maxsplit=1)[0]

        if tag == "B":
            if current_span:
                spans.append(current_span)
            current_span = {
                "start": start_char,
                "end": end_char,
                "text": text[start_char:end_char]
            }

        elif tag == "I":
            if current_span:
                current_span["end"] = end_char
                current_span["text"] = text[current_span["start"]:end_char]
            else:
                # I without B (improper tagging) → treat as B
                current_span = {
                    "start": start_char,
                    "end": end_char,
                    "text": text[start_char:end_char]
                }

    if current_span:
        spans.append(current_span)

    return spans
