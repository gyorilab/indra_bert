from transformers import AutoTokenizer
import torch
from pathlib import Path
from .preprocess import preprocess_for_inference, preprocess_for_inference_batch
from .bert_classification_head import TwoGatedClassifier


class IndraStmtClassifier:
    def __init__(self, model_path, device=None):
        model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model = TwoGatedClassifier.from_pretrained(model_path)
            
        self.model.eval()

        # Device (auto detect)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model.to(self.device)
        self.id2label = self.model.config.id2label

    def predict(self, text):
        enc = preprocess_for_inference(
            text=text,
            tokenizer=self.tokenizer,
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            # Use two-gated prediction
            outputs = self.model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predicted_class = outputs['predictions'][0]
            confidence = outputs['confidences'][0]
            
            # Handle -1 (no relation) predictions
            if predicted_class == -1:
                predicted_label = "No_Relation"
            else:
                predicted_label = self.id2label[predicted_class]
            
            # Create probability distribution from gate outputs
            gate2_probs = outputs['gate2_probs'][0]
            prob_dist = {self.id2label[i]: gate2_probs[i].item() for i in range(len(self.id2label))}

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": prob_dist,
            "input_ids": enc["input_ids"],
            "decoded_text": self.tokenizer.decode(enc["input_ids"].squeeze()),
            "original_text": text
        }

    def predict_batch(self, texts: list[str]):
        assert isinstance(texts, list) and len(texts) > 0, "Input must be a non-empty list of strings."
        assert all(isinstance(text, str) for text in texts), "All elements in the input list must be strings."

        # Step 1: Batch tokenization
        enc = preprocess_for_inference_batch(
            texts=texts,
            tokenizer=self.tokenizer
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Step 2: Model inference
        with torch.no_grad():
            # Use two-gated prediction
            outputs = self.model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predicted_classes = outputs['predictions']
            confidences = outputs['confidences']
            gate2_probs = outputs['gate2_probs']

        results = []
        for i in range(len(texts)):
            # Handle -1 (no relation) predictions
            if predicted_classes[i] == -1:
                predicted_label = "No_Relation"
            else:
                predicted_label = self.id2label[predicted_classes[i]]
                
            confidence = confidences[i]
            prob_dist = {self.id2label[j]: gate2_probs[i][j].item() for j in range(len(self.id2label))}

            results.append({
                "predicted_label": predicted_label,
                "confidence": confidence,
                "probabilities": prob_dist,
                "input_ids": input_ids[i],
                "decoded_text": self.tokenizer.decode(input_ids[i]),
                "original_text": texts[i]            
            })

        return results
