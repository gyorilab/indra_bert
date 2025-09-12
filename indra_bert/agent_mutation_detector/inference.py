from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
from typing import List, Dict, Tuple, Any

class AgentMutationDetector:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label
        
        self.model.eval()

    def predict(self, pair: List[Dict], annotated_text: str) -> Dict[str, Any]:
        """
        Predict mutations for each agent in the pair.
        
        Args:
            pair: List of agent entities [{"start": 27, "end": 32, "text": "Hsp70"}, ...]
            annotated_text: Text with all agents marked with <e></e> tags
            
        Returns:
            Dict with mutations keyed by (start, end, text) tuples
        """
        mutations_dict = {}
        
        for agent in pair:
            # Create modified text with only current agent tagged
            modified_text = self._create_single_agent_text(annotated_text, agent)
            
            # Run mutation detection on this agent
            mutation_spans = self._detect_mutations_for_agent(modified_text)
            
            if mutation_spans:
                # Convert mutation spans to mutation objects
                mutations = self._spans_to_mutations(mutation_spans, modified_text)
                agent_key = (agent["start"], agent["end"], agent["text"])
                mutations_dict[agent_key] = mutations
        
        return {
            "mutations": mutations_dict,
            "raw_output": {
                "pair": pair,
                "annotated_text": annotated_text,
                "mutations_dict": mutations_dict
            }
        }

    def predict_batch(self, pairs: List[List[Dict]], annotated_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict mutations for multiple pairs in batch.
        
        Args:
            pairs: List of agent pairs
            annotated_texts: List of annotated texts corresponding to each pair
            
        Returns:
            List of mutation prediction results
        """
        results = []
        for pair, annotated_text in zip(pairs, annotated_texts):
            result = self.predict(pair, annotated_text)
            results.append(result)
        return results

    def _create_single_agent_text(self, annotated_text: str, target_agent: Dict) -> str:
        """
        Create text with only the target agent tagged with <e></e>.
        
        Args:
            annotated_text: Original text with all agents tagged
            target_agent: The agent to keep tagged
            
        Returns:
            Modified text with only target agent tagged
        """
        # Remove all <e></e> tags first
        clean_text = re.sub(r'<e>(.*?)</e>', r'\1', annotated_text)
        
        # Add back only the target agent's tags
        agent_text = target_agent["text"]
        start = target_agent["start"]
        end = target_agent["end"]
        
        # Insert tags around the target agent
        modified_text = clean_text[:start] + f"<e>{agent_text}</e>" + clean_text[end:]
        
        return modified_text

    def _detect_mutations_for_agent(self, text: str) -> List[Dict]:
        """
        Detect mutation spans in text for a specific agent.
        
        Args:
            text: Text with only the target agent tagged
            
        Returns:
            List of mutation spans
        """
        # Tokenize and run model
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        model_inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
        
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze(0).tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze(0))
        offset_mapping = encoding["offset_mapping"].squeeze(0)
        
        # Extract mutation spans from predictions
        mutation_spans = self._extract_mutation_spans(tokens, offset_mapping, predictions, text)
        
        return mutation_spans

    def _extract_mutation_spans(self, tokens: List[str], offset_mapping: List[Tuple], 
                               predictions: List[int], original_text: str) -> List[Dict]:
        """
        Extract mutation spans from token-level predictions.
        
        Args:
            tokens: List of tokens
            offset_mapping: Token to character position mapping
            predictions: Predicted label IDs for each token
            original_text: Original text
            
        Returns:
            List of mutation span dictionaries
        """
        mutation_spans = []
        current_span = None
        
        for i, (token, offset, pred_id) in enumerate(zip(tokens, offset_mapping, predictions)):
            if offset[0] == offset[1]:  # Skip special tokens
                continue
                
            label = self.id2label.get(pred_id, "O")
            
            if label.startswith("B-mutation"):
                # Start new mutation span
                if current_span:
                    mutation_spans.append(current_span)
                current_span = {
                    "start": offset[0],
                    "end": offset[1],
                    "text": original_text[offset[0]:offset[1]]
                }
            elif label.startswith("I-mutation") and current_span:
                # Continue current mutation span
                current_span["end"] = offset[1]
                current_span["text"] = original_text[current_span["start"]:offset[1]]
            else:
                # End current span if exists
                if current_span:
                    mutation_spans.append(current_span)
                    current_span = None
        
        # Add final span if exists
        if current_span:
            mutation_spans.append(current_span)
        
        return mutation_spans

    def _spans_to_mutations(self, mutation_spans: List[Dict], text: str) -> List[Dict]:
        """
        Convert mutation spans to mutation objects.
        
        Args:
            mutation_spans: List of mutation spans
            text: Original text
            
        Returns:
            List of mutation dictionaries
        """
        mutations = []
        
        for span in mutation_spans:
            mutation_text = span["text"]
            
            # Try to parse mutation format (e.g., "E123L", "K77R")
            mutation_info = self._parse_mutation_text(mutation_text)
            if mutation_info:
                mutations.append(mutation_info)
            else:
                # If parsing fails, create a generic mutation
                mutations.append({
                    "position": "unknown",
                    "residue_from": "unknown", 
                    "residue_to": "unknown",
                    "text": mutation_text
                })
        
        return mutations

    def _parse_mutation_text(self, mutation_text: str) -> Dict:
        """
        Parse mutation text to extract position and residue changes.
        
        Args:
            mutation_text: Text like "E123L", "K77R", etc.
            
        Returns:
            Dictionary with parsed mutation info or None if parsing fails
        """
        # Pattern for single letter mutations: E123L, K77R, etc.
        pattern = r'^([A-Z])(\d+)([A-Z])$'
        match = re.match(pattern, mutation_text)
        
        if match:
            residue_from = match.group(1)
            position = match.group(2)
            residue_to = match.group(3)
            
            return {
                "position": position,
                "residue_from": residue_from,
                "residue_to": residue_to,
                "text": mutation_text
            }
        
        return None
