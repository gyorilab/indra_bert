from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from ner_agent_detector.preprocess import preprocess_for_inference
from ner_agent_detector.postprocess import extract_spans_from_encoding

class AgentNERModel:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label

        self.model.eval()

    def predict_raw(self, text: str):
        encoding = preprocess_for_inference(text, self.tokenizer)

        model_inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
        
        with torch.no_grad():
            outputs = self.model(**model_inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze(0).tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze(0))
        offset_mapping = encoding["offset_mapping"]

        return tokens, offset_mapping, predictions

    def predict(self, text: str):
        tokens, offset_mapping, predicted_label_ids = self.predict_raw(text)
        entity_spans = extract_spans_from_encoding(tokens, offset_mapping, predicted_label_ids, self.id2label, text)
        annotated_text = self._annotate_text(text, entity_spans)
        return {
            "text": text,
            "entity_spans": entity_spans,
            "annotated_text": annotated_text
        }

    def _annotate_text(self, text, entity_spans):
        # Insert <e> tags around detected spans for visualization
        entity_spans_sorted = sorted(entity_spans, key=lambda x: x["start"], reverse=True)
        for span in entity_spans_sorted:
            start = span["start"]
            end = span["end"]
            entity_text = span["text"]
            text = text[:start] + f"<e>{entity_text}</e>" + text[end:]
        return text
