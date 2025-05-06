from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pathlib import Path
from typing import List


class NERAnnotator:
    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def annotate(self, text: str) -> dict:
        """
        Annotate a single text with entity tags.

        Returns:
            {
                "annotated_text": ...,
                "bio_tags": [...]
            }
        """
        encoding = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = encoding.pop("offset_mapping")[0]
        input_ids = encoding["input_ids"][0]

        with torch.no_grad():
            outputs = self.model(**encoding)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0]
        pred_labels = [self.id2label[p.item()] for p in predictions]

        annotated_text = ""
        inside_agent = False
        last_end = 0

        for (offset, label) in zip(offsets.tolist(), pred_labels):
            start, end = offset

            # Skip special tokens
            if start == end:
                continue

            word = text[start:end]

            # Add space if necessary
            if start > last_end:
                annotated_text += text[last_end:start]

            # Annotation logic
            if label == "B-AGENT":
                if inside_agent:
                    annotated_text += "</Agent>"
                annotated_text += "<Agent>" + word
                inside_agent = True
            elif label == "I-AGENT":
                annotated_text += word
            else:
                if inside_agent:
                    annotated_text += "</Agent>"
                    inside_agent = False
                annotated_text += word

            last_end = end

        if inside_agent:
            annotated_text += "</Agent>"

        return {
            "annotated_text": annotated_text,
            "tokens": self.tokenizer.convert_ids_to_tokens(input_ids),
            "bio_tags": pred_labels
        }


# Example usage (optional)
if __name__ == "__main__":
    annotator = NERAnnotator("output/ner_model/checkpoint-176")

    text = "The fundamental abnormality resulting in the development of cancer is the continual unregulated proliferation of cancer cells."
    annotated_text = annotator.annotate(text)

    print("Annotated Text:")
    print(annotated_text)
