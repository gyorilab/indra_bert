"""
Inference script for two-step negation detection.
1. Detect all cues
2. For each cue, predict its scope
"""
import json
import torch
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoConfig

from .model import NegationDetectorModel
from .preprocess import char_to_token_position


class NegationDetector:
    """
    Two-step negation detector:
    1. Detects all negation cues
    2. For each cue, predicts its scope span
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the negation detector.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load label mappings
        label_path = self.model_path / "cue_label2id.json"
        if not label_path.exists():
            label_path = self.model_path.parent / "cue_label2id.json"
        
        if label_path.exists():
            with open(label_path, "r") as f:
                self.cue_label2id = json.load(f)
        else:
            # Fallback: try cached_dataset
            cached_label_path = self.model_path.parent / "cached_dataset" / "cue_label2id.json"
            if cached_label_path.exists():
                with open(cached_label_path, "r") as f:
                    self.cue_label2id = json.load(f)
            else:
                raise FileNotFoundError(f"Could not find cue_label2id.json")
        
        self.cue_id2label = {v: k for k, v in self.cue_label2id.items()}
        
        # Load model config
        config = AutoConfig.from_pretrained(model_path)
        config.cue_num_labels = len(self.cue_label2id)
        
        # Initialize model
        self.model = NegationDetectorModel.from_pretrained(
            model_path,
            config=config,
            cue_num_labels=len(self.cue_label2id),
        )
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict negation cues and scopes for a single text.
        
        Args:
            text: Input text string
        
        Returns:
            Dictionary with:
            - 'text': Original text
            - 'cues': List of cue spans [{'start': int, 'end': int, 'text': str}, ...]
            - 'scopes': List of scope spans [{'start': int, 'end': int, 'text': str}, ...]
            - 'negations': List of paired negations [{'cue': {...}, 'scope': {...}}, ...]
        """
        # Step 1: Detect all cues
        cues = self._detect_cues(text)
        
        # Step 2: For each cue, predict its scope
        scopes = []
        negations = []
        
        for cue in cues:
            scope = self._predict_scope(text, cue)
            if scope:
                scopes.append(scope)
                negations.append({
                    "cue": cue,
                    "scope": scope,
                })
            else:
                negations.append({
                    "cue": cue,
                    "scope": None,
                })
        
        return {
            "text": text,
            "cues": cues,
            "scopes": scopes,
            "negations": negations,
        }
    
    def _detect_cues(self, text: str) -> List[Dict[str, Any]]:
        """Detect all negation cues in the text."""
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
            return_offsets_mapping=True,
        )
        
        model_inputs = {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }
        if "token_type_ids" in encoding:
            model_inputs["token_type_ids"] = encoding["token_type_ids"].to(self.device)
        
        # Predict cues
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        
        cue_logits = outputs.logits  # [1, L, cue_num_labels]
        cue_preds = torch.argmax(cue_logits, dim=-1).squeeze(0).cpu().tolist()
        
        # Extract cue spans
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze(0))
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()
        
        cues = []
        current_span = None
        
        for i, label_id in enumerate(cue_preds):
            if i >= len(offset_mapping):
                break
            
            label = self.cue_id2label.get(label_id, "O")
            tok_start, tok_end = offset_mapping[i]
            
            if tok_start is None or tok_end is None:
                continue
            
            if label == "B-cue":
                if current_span:
                    cues.append(current_span)
                current_span = {
                    "start": tok_start,
                    "end": tok_end,
                    "text": text[tok_start:tok_end],
                }
            elif label == "I-cue":
                if current_span:
                    current_span["end"] = tok_end
                    current_span["text"] = text[current_span["start"]:tok_end]
            else:
                if current_span:
                    cues.append(current_span)
                    current_span = None
        
        if current_span:
            cues.append(current_span)
        
        return cues
    
    def _predict_scope(self, text: str, cue: Dict[str, Any]) -> Dict[str, Any]:
        """Predict scope span for a given cue."""
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
            return_offsets_mapping=True,
        )
        
        # Find cue token position
        cue_char_start = cue["start"]
        cue_token_position = char_to_token_position(
            encoding["offset_mapping"].squeeze(0).tolist(),
            cue_char_start
        )
        
        # Ensure valid position
        seq_len = len(encoding["offset_mapping"].squeeze(0))
        cue_token_position = max(0, min(cue_token_position, seq_len - 1))
        
        model_inputs = {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
            "cue_positions": torch.tensor([cue_token_position], dtype=torch.long).to(self.device),
        }
        if "token_type_ids" in encoding:
            model_inputs["token_type_ids"] = encoding["token_type_ids"].to(self.device)
        
        # Predict scope
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        
        scope_start_logits = outputs.scope_start_logits  # [1, L]
        scope_end_logits = outputs.scope_end_logits      # [1, L]
        
        # Mask invalid positions
        attention_mask = model_inputs["attention_mask"].bool()
        masked_start_logits = scope_start_logits.masked_fill(~attention_mask, float('-inf'))
        masked_end_logits = scope_end_logits.masked_fill(~attention_mask, float('-inf'))
        
        # Predict positions
        scope_start_token = torch.argmax(masked_start_logits, dim=-1).item()
        scope_end_token = torch.argmax(masked_end_logits, dim=-1).item()
        
        # Ensure valid span: end_token must be >= start_token
        if scope_end_token < scope_start_token:
            # If model predicts invalid span, swap them or use start_token as end
            scope_end_token = scope_start_token
        
        # Convert token positions to character positions
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()
        
        if scope_start_token < len(offset_mapping) and scope_end_token < len(offset_mapping):
            start_char, _ = offset_mapping[scope_start_token]
            _, end_char = offset_mapping[scope_end_token]
            
            if start_char is not None and end_char is not None:
                # Double-check: ensure start_char <= end_char
                if start_char <= end_char:
                    return {
                        "start": start_char,
                        "end": end_char,
                        "text": text[start_char:end_char],
                    }
                else:
                    # If still invalid, return None (model prediction is too poor)
                    return None
        
        return None
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict negation cues and scopes for a batch of texts.
        
        Args:
            texts: List of input text strings
        
        Returns:
            List of prediction dictionaries (same format as predict)
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

