import json
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoConfig

from .model import MultiHeadStmtClassifier
from .preprocess import (
    preprocess_for_inference,
    preprocess_for_inference_batch
)


class IndraStmtClassifier:
    def __init__(self, model_path: str | Path, device: Optional[torch.device] = None):
        self.model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if "<e>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<e>", "</e>"]})

        binary_label_map_path = self.model_path / "binary_label2id.json"
        relation_map_path = self.model_path / "relation_subtype2id.json"
        indra_map_path = self.model_path / "indra_label2id.json"
        if not binary_label_map_path.exists() or not relation_map_path.exists() or not indra_map_path.exists():
            raise FileNotFoundError("Label mapping files not found in model directory.")

        with binary_label_map_path.open("r", encoding="utf-8") as fh:
            self.binary_label2id = {str(k): int(v) for k, v in json.load(fh).items()}
        with relation_map_path.open("r", encoding="utf-8") as fh:
            self.relation_subtype2id = json.load(fh)
        with indra_map_path.open("r", encoding="utf-8") as fh:
            self.indra_label2id = json.load(fh)

        self.id2binary_label = {v: k for k, v in self.binary_label2id.items()}
        self.id2relation_subtype = {v: k for k, v in self.relation_subtype2id.items()}
        self.id2indra_label = {v: k for k, v in self.indra_label2id.items()}

        config = AutoConfig.from_pretrained(self.model_path)
        self.model = MultiHeadStmtClassifier.from_pretrained(
            self.model_path,
            config=config,
            gate2_num_labels=len(self.relation_subtype2id),
            gate3_num_labels=len(self.indra_label2id),
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def _format_output(self, batch_output, threshold: float) -> List[dict]:
        gate1_preds = batch_output["gate1_predictions"].cpu().tolist()
        gate1_probs = batch_output["gate1_probs"].cpu().tolist()
        gate2_preds = batch_output["gate2_predictions"].cpu().tolist()
        gate2_probs = batch_output["gate2_probs"].cpu().tolist()
        gate3_preds = batch_output["gate3_predictions"].cpu().tolist()
        gate3_probs = batch_output["gate3_probs"].cpu().tolist()

        results = []
        for g1_pred, g1_prob, g2_pred, g2_prob, g3_pred, g3_prob in zip(
            gate1_preds, gate1_probs, gate2_preds, gate2_probs, gate3_preds, gate3_probs
        ):
            gate1_label = self.id2binary_label.get(g1_pred)
            subtype_label = self.id2relation_subtype.get(g2_pred, "unknown")
            indra_label = self.id2indra_label.get(g3_pred, "unknown")
            gate1_prob_map = {
                self.id2binary_label.get(idx, str(idx)): prob 
                for idx, prob in enumerate(g1_prob)
            }
            gate2_prob_map = {
                self.id2relation_subtype.get(idx, str(idx)): prob
                for idx, prob in enumerate(g2_prob)
            }
            gate3_prob_map = {
                self.id2indra_label.get(idx, str(idx)): prob
                for idx, prob in enumerate(g3_prob)
            }
            results.append(
                {
                    "gate1_prediction": gate1_label,
                    "gate1_probs": gate1_prob_map,
                    "gate2_prediction": subtype_label,
                    "gate2_probs": gate2_prob_map,
                    "gate3_prediction": indra_label,
                    "gate3_probs": gate3_prob_map,
                }
            )
        return results

    def predict(self, text: str, threshold: float = 0.5) -> dict:
        enc = preprocess_for_inference(text, self.tokenizer)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        output = self.model.predict(**enc, threshold=threshold)
        return self._format_output(output, threshold)[0]

    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> List[dict]:
        enc = preprocess_for_inference_batch(texts, self.tokenizer)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        output = self.model.predict(**enc, threshold=threshold)
        return self._format_output(output, threshold)
