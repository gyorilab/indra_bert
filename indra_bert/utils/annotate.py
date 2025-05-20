def annotate_entities(text, entity_spans):
    # Insert <e> tags around detected spans for visualization
    entity_spans_sorted = sorted(entity_spans, key=lambda x: x["start"], reverse=True)
    for span in entity_spans_sorted:
        start = span["start"]
        end = span["end"]
        entity_text = span["text"]
        text = text[:start] + f"<e>{entity_text}</e>" + text[end:]
    return text
