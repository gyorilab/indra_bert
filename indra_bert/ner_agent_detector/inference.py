from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from .preprocess import preprocess_for_inference
from .postprocess import extract_spans_from_encoding

class AgentNERExtractor:
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

    def predict_batch(self, texts: list[str]):
        # Step 1: Tokenize as a batch
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            max_length=512,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        offset_mappings = encodings["offset_mapping"]  # shape: (batch_size, seq_len, 2)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (batch_size, seq_len, num_labels)
            predictions = torch.argmax(logits, dim=2)  # shape: (batch_size, seq_len)

        results = []
        for i in range(len(texts)):
            token_ids = input_ids[i]
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            preds = predictions[i].tolist()
            offsets = offset_mappings[i].tolist()

            entity_spans = extract_spans_from_encoding(
                tokens=tokens,
                offset_mapping=offsets,
                predicted_label_ids=preds,
                id2label=self.id2label,
                text=texts[i]
            )

            annotated_text = self._annotate_text(texts[i], entity_spans)

            results.append({
                "text": texts[i],
                "entity_spans": entity_spans,
                "annotated_text": annotated_text
            })

        return results
