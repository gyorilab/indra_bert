import json
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

from .model import MultiHeadStmtClassifier
from .preprocess import (
    preprocess_for_inference,
    preprocess_for_inference_batch,
)


class IndraStmtClassifier:
    def __init__(self, model_path: str | Path, device: Optional[torch.device] = None):
        candidate_path = Path(model_path)
        if candidate_path.exists():
            self.model_path = candidate_path.resolve()
            pretrained_source = str(self.model_path)
        else:
            snapshot_path = snapshot_download(repo_id=str(model_path), repo_type="model")
            self.model_path = Path(snapshot_path)
            pretrained_source = snapshot_path

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_source)
        if "<e>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<e>", "</e>"]})

        binary_label_map_path = self.model_path / "binary_label2id.json"
        relation_map_path = self.model_path / "relation_subtype2id.json"
        indra_map_path = self.model_path / "indra_label2id.json"
        gate4_label_map_path = self.model_path / "gate4_label2id.json"
        
        if not binary_label_map_path.exists() or not relation_map_path.exists() or not indra_map_path.exists():
            raise FileNotFoundError("Label mapping files not found in model directory.")

        with binary_label_map_path.open("r", encoding="utf-8") as fh:
            self.binary_label2id = {str(k): int(v) for k, v in json.load(fh).items()}
        with relation_map_path.open("r", encoding="utf-8") as fh:
            self.relation_subtype2id = json.load(fh)
        with indra_map_path.open("r", encoding="utf-8") as fh:
            self.indra_label2id = json.load(fh)
        
        # Load gate4 label mappings if available
        self.gate4_label2id = None
        self.gate4_id2label = None
        if gate4_label_map_path.exists():
            with gate4_label_map_path.open("r", encoding="utf-8") as fh:
                self.gate4_label2id = json.load(fh)
                self.gate4_id2label = {v: k for k, v in self.gate4_label2id.items()}

        self.id2binary_label = {v: k for k, v in self.binary_label2id.items()}
        self.id2relation_subtype = {v: k for k, v in self.relation_subtype2id.items()}
        self.id2indra_label = {v: k for k, v in self.indra_label2id.items()}

        config = AutoConfig.from_pretrained(pretrained_source)
        gate4_num_labels = len(self.gate4_label2id) if self.gate4_label2id else 3
        self.model = MultiHeadStmtClassifier.from_pretrained(
            pretrained_source,
            config=config,
            gate2_num_labels=len(self.relation_subtype2id),
            gate3_num_labels=len(self.indra_label2id),
            gate4_num_labels=gate4_num_labels,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def _extract_predicate_spans(self, gate4_predictions, offset_mapping, original_text, input_text_with_tags):
        """
        Extract predicate spans from gate4 token predictions.
        
        Important: BIO tags are assigned based on token positions in the tagged input text
        (with <e> tags), but output spans are re-indexed to character positions in the
        original untagged text (without <e> tags).
        
        Args:
            gate4_predictions: [L] tensor or list of predicted label IDs (BIO tags for tagged text)
            offset_mapping: [(start, end), ...] token offset mappings in input_text_with_tags
            original_text: Original text without any tags (for output span positions)
            input_text_with_tags: Input text with <e> tags (used for tokenization and BIO labels)
        
        Returns:
            List of dicts with 'start', 'end', 'text' keys (character positions in original_text)
        """
        if self.gate4_id2label is None:
            return []
        
        import re
        
        spans = []
        current_span = None
        
        # Build mapping from input_text_with_tags character positions to original_text character positions
        # This re-indexes spans from the tagged text (where BIO labels are assigned) to the
        # original untagged text (where output spans should be reported)
        # Skip <e> and </e> tags when mapping
        tag_to_orig_map = {}
        orig_idx = 0
        input_idx = 0
        
        while orig_idx < len(original_text) and input_idx < len(input_text_with_tags):
            if input_text_with_tags[input_idx] == '<':
                # Skip tag characters
                while input_idx < len(input_text_with_tags) and input_text_with_tags[input_idx] != '>':
                    input_idx += 1
                if input_idx < len(input_text_with_tags):
                    input_idx += 1
            else:
                # Map: character position in tagged text -> character position in original text
                tag_to_orig_map[input_idx] = orig_idx
                orig_idx += 1
                input_idx += 1
        
        # Extract spans from token predictions
        for i, label_id in enumerate(gate4_predictions):
            if i >= len(offset_mapping):
                break
                
            label = self.gate4_id2label.get(label_id.item() if torch.is_tensor(label_id) else label_id, "O")
            tok_start, tok_end = offset_mapping[i]
            
            if tok_start is None or tok_end is None:
                continue
            
            if label == "B-trigger":
                # Start of new span
                if current_span is not None:
                    spans.append(current_span)
                # Map token positions (in tagged text) to character positions (in original untagged text)
                # tok_start, tok_end are character positions in input_text_with_tags
                # We need to map them to character positions in original_text
                orig_start = tag_to_orig_map.get(tok_start, tok_start)
                # For end position, try tok_end-1 first (since tok_end is exclusive)
                if (tok_end - 1) in tag_to_orig_map:
                    orig_end = tag_to_orig_map[tok_end - 1] + 1
                elif tok_end in tag_to_orig_map:
                    orig_end = tag_to_orig_map[tok_end]
                else:
                    orig_end = tok_end  # Fallback
                
                current_span = {
                    "start": orig_start,
                    "end": orig_end,
                    "text": original_text[orig_start:orig_end] if orig_start < len(original_text) and orig_end <= len(original_text) else "",
                }
            elif label == "I-trigger":
                # Continue current span
                if current_span is not None:
                    # Map token end position to original text
                    if (tok_end - 1) in tag_to_orig_map:
                        orig_end = tag_to_orig_map[tok_end - 1] + 1
                    elif tok_end in tag_to_orig_map:
                        orig_end = tag_to_orig_map[tok_end]
                    else:
                        orig_end = tok_end  # Fallback
                    
                    current_span["end"] = orig_end
                    if current_span["start"] < len(original_text) and orig_end <= len(original_text):
                        current_span["text"] = original_text[current_span["start"]:orig_end]
            else:
                # O - end current span
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
        
        # Don't forget the last span
        if current_span is not None:
            spans.append(current_span)
        
        return spans

    def _format_output(self, batch_output, threshold: float, texts: List[str] = None, encodings=None) -> List[dict]:
        gate1_preds = batch_output["gate1_predictions"].cpu().tolist()
        gate1_probs = batch_output["gate1_probs"].cpu().tolist()
        gate2_preds = batch_output["gate2_predictions"].cpu().tolist()
        gate2_probs = batch_output["gate2_probs"].cpu().tolist()
        gate3_preds = batch_output["gate3_predictions"].cpu().tolist()
        gate3_probs = batch_output["gate3_probs"].cpu().tolist()
        gate4_preds = batch_output.get("gate4_predictions")  # [B, L] tensor

        results = []
        for i, (g1_pred, g1_prob, g2_pred, g2_prob, g3_pred, g3_prob) in enumerate(zip(
            gate1_preds, gate1_probs, gate2_preds, gate2_probs, gate3_preds, gate3_probs
        )):
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
            
            result = {
                "gate1_prediction": gate1_label,
                "gate1_probs": gate1_prob_map,
                "gate2_prediction": subtype_label,
                "gate2_probs": gate2_prob_map,
                "gate3_prediction": indra_label,
                "gate3_probs": gate3_prob_map,
            }
            
            # Extract predicate spans from gate4 if available
            if gate4_preds is not None and self.gate4_id2label is not None and texts is not None and encodings is not None:
                import re
                # Get original text (remove <e> tags)
                original_text = re.sub(r'<e>|</e>', '', texts[i])
                # Get input text with tags (for offset mapping)
                input_text_with_tags = texts[i]
                # Get offset mapping for this example
                offset_mapping = encodings["offset_mapping"][i].cpu().tolist()
                # Get gate4 predictions for this example
                gate4_pred = gate4_preds[i]  # [L] tensor
                
                predicate_spans = self._extract_predicate_spans(
                    gate4_pred, offset_mapping, original_text, input_text_with_tags
                )
                result["gate4_predicate_spans"] = predicate_spans
            
            results.append(result)
        return results

    def predict(self, text: str, threshold: float = 0.5) -> dict:
        enc = preprocess_for_inference(text, self.tokenizer)
        # Get normalized text for span extraction (needed to map back to original)
        from .preprocess.shared import normalize_entity_tags
        text_normalized = normalize_entity_tags(text)
        enc_for_model = {k: v.to(self.device) for k, v in enc.items() if k != "offset_mapping"}
        output = self.model.predict(**enc_for_model, threshold=threshold)
        return self._format_output(output, threshold, texts=[text_normalized], encodings=enc)[0]

    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> List[dict]:
        enc = preprocess_for_inference_batch(texts, self.tokenizer)
        # Get normalized texts for span extraction (needed to map back to original)
        from .preprocess.shared import normalize_entity_tags
        texts_normalized = [normalize_entity_tags(t) for t in texts]
        enc_for_model = {k: v.to(self.device) for k, v in enc.items() if k != "offset_mapping"}
        output = self.model.predict(**enc_for_model, threshold=threshold)
        return self._format_output(output, threshold, texts=texts_normalized, encodings=enc)
