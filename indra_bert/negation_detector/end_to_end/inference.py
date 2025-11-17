"""
Inference script for negation detection.
Extracts negation cues and scopes from text.
"""
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoConfig

from .model import NegationDetectorModel
from .postprocess import extract_spans_from_encoding


class NegationDetector:
    """
    Negation detector that extracts negation cues and scopes from text.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the negation detector.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load label mappings (check both model_path and parent directory)
        label_path = self.model_path / "cue_label2id.json"
        if not label_path.exists():
            # Try parent directory (for checkpoint directories)
            label_path = self.model_path.parent / "cue_label2id.json"
        
        if label_path.exists():
            with open(label_path, "r") as f:
                self.cue_label2id = json.load(f)
        else:
            # Fallback: try cached_dataset directory
            cached_label_path = self.model_path.parent / "cached_dataset" / "cue_label2id.json"
            if cached_label_path.exists():
                with open(cached_label_path, "r") as f:
                    self.cue_label2id = json.load(f)
            else:
                raise FileNotFoundError(f"Could not find cue_label2id.json in {self.model_path} or parent directories")
        
        scope_label_path = self.model_path / "scope_label2id.json"
        if not scope_label_path.exists():
            scope_label_path = self.model_path.parent / "scope_label2id.json"
        
        if scope_label_path.exists():
            with open(scope_label_path, "r") as f:
                self.scope_label2id = json.load(f)
        else:
            cached_scope_label_path = self.model_path.parent / "cached_dataset" / "scope_label2id.json"
            if cached_scope_label_path.exists():
                with open(cached_scope_label_path, "r") as f:
                    self.scope_label2id = json.load(f)
            else:
                raise FileNotFoundError(f"Could not find scope_label2id.json in {self.model_path} or parent directories")
        
        self.cue_id2label = {v: k for k, v in self.cue_label2id.items()}
        self.scope_id2label = {v: k for k, v in self.scope_label2id.items()}
        
        # Load model config
        config = AutoConfig.from_pretrained(model_path)
        config.cue_num_labels = len(self.cue_label2id)
        config.scope_num_labels = len(self.scope_label2id)
        
        # Initialize model
        self.model = NegationDetectorModel.from_pretrained(
            model_path,
            config=config,
            cue_num_labels=len(self.cue_label2id),
            scope_num_labels=len(self.scope_label2id),
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
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
            return_offsets_mapping=True,
        )
        
        # Move to device
        model_inputs = {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }
        if "token_type_ids" in encoding:
            model_inputs["token_type_ids"] = encoding["token_type_ids"].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        
        cue_logits = outputs.logits  # [1, L, cue_num_labels]
        scope_logits = outputs.scope_logits  # [1, L, scope_num_labels]
        
        cue_preds = torch.argmax(cue_logits, dim=-1).squeeze(0).cpu().tolist()
        scope_preds = torch.argmax(scope_logits, dim=-1).squeeze(0).cpu().tolist()
        
        # Extract spans
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze(0))
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()
        
        cues = extract_spans_from_encoding(
            tokens, offset_mapping, cue_preds, self.cue_id2label, text, span_type="cue"
        )
        scopes = extract_spans_from_encoding(
            tokens, offset_mapping, scope_preds, self.scope_id2label, text, span_type="scope"
        )
        
        # Post-process scopes to extend outer scopes when they contain inner scopes
        from .postprocess import extend_outer_scopes
        scopes = extend_outer_scopes(scopes)
        
        # Update scope text after extension
        for scope in scopes:
            if scope['start'] < len(text) and scope['end'] <= len(text):
                scope['text'] = text[scope['start']:scope['end']]
        
        # Pair cues with scopes (simple heuristic: match each cue with the nearest scope)
        negations = self._pair_cues_scopes(cues, scopes)
        
        return {
            "text": text,
            "cues": cues,
            "scopes": scopes,
            "negations": negations,
        }
    
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
    
    def _pair_cues_scopes(
        self,
        cues: List[Dict[str, Any]],
        scopes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Pair cues with scopes using a simple heuristic:
        - Match each cue with the nearest scope that starts after the cue
        - If no scope starts after the cue, match with the nearest scope overall
        
        Args:
            cues: List of cue spans
            scopes: List of scope spans
        
        Returns:
            List of paired negations [{'cue': {...}, 'scope': {...}}, ...]
        """
        negations = []
        
        for cue in cues:
            cue_start = cue["start"]
            cue_end = cue["end"]
            
            # Find the nearest scope that starts after or overlaps with the cue
            best_scope = None
            min_distance = float('inf')
            
            for scope in scopes:
                scope_start = scope["start"]
                scope_end = scope["end"]
                
                # Prefer scopes that start after or overlap with the cue
                if scope_start >= cue_start:
                    distance = scope_start - cue_start
                    if distance < min_distance:
                        min_distance = distance
                        best_scope = scope
                elif scope_end > cue_start:
                    # Overlapping scope
                    distance = 0
                    if distance < min_distance:
                        min_distance = distance
                        best_scope = scope
            
            # If no scope found after cue, use the nearest scope overall
            if best_scope is None and scopes:
                for scope in scopes:
                    distance = abs(scope["start"] - cue_start)
                    if distance < min_distance:
                        min_distance = distance
                        best_scope = scope
            
            if best_scope:
                negations.append({
                    "cue": cue,
                    "scope": best_scope,
                })
            else:
                # Cue without matching scope
                negations.append({
                    "cue": cue,
                    "scope": None,
                })
        
        return negations

