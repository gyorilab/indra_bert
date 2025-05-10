from transformers import AutoTokenizer
import torch
from pathlib import Path
from indra_stmt_classifier_model.preprocess import preprocess_for_inference
from indra_stmt_classifier_model.relation_model import BertForIndraStmtClassification


class IndraStmtClassifier:
    def __init__(self, model_path, device=None):
        model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BertForIndraStmtClassification.from_pretrained(model_path)
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
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted_class = probs.max(dim=-1)
            predicted_label = self.id2label[int(predicted_class.item())]

        prob_dist = {label: probs[0][i].item() for i, label in enumerate(self.id2label.values())}

        return {
            "predicted_label": predicted_label,
            "confidence": confidence.item(),
            "probabilities": prob_dist,
            "input_ids": enc["input_ids"],
            "decoded_text": self.tokenizer.decode(enc["input_ids"].squeeze())
        }
