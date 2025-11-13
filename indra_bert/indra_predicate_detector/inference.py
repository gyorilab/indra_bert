from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import List, Dict, Any
import re

from .postprocess import extract_trigger_spans_from_encoding


class PredicateDetector:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Add special tokens (idempotent - won't add if already present)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<e>", "</e>"]})
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict trigger spans and roles from text with <e>entity</e> tags.
        
        Args:
            text: Input text with <e>entity</e> tags (and optionally <trigger> tags which will be removed)
        
        Returns:
            Dictionary with:
            - triggers: List of trigger spans with start, end, text, role
            - annotated_text: Text with predicted trigger tags
        """
        # Remove any existing <trigger> tags from input
        input_text = re.sub(r'</?trigger>', '', text)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].cpu().tolist()

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().tolist()

        # Extract tokens and spans
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        trigger_spans = extract_trigger_spans_from_encoding(
            tokens, offset_mapping, predictions, self.id2label, input_text
        )

        # Create annotated text with predicted triggers
        annotated_text = input_text
        # Insert trigger tags in reverse order to preserve positions
        for span in sorted(trigger_spans, key=lambda x: x["start"], reverse=True):
            trigger_tag = f'<trigger role="{span["role"]}">{span["text"]}</trigger>'
            annotated_text = (
                annotated_text[: span["start"]]
                + trigger_tag
                + annotated_text[span["end"] :]
            )

        return {
            "triggers": trigger_spans,
            "annotated_text": annotated_text,
            "text": input_text,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict triggers for a batch of texts."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

