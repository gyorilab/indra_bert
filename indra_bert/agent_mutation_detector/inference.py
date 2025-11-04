from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
from typing import List, Dict, Tuple, Any

class AgentMutationDetector:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Add special tokens to match training setup
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<e>', '</e>']})
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, agents: List[Dict], annotated_text: str) -> Dict[str, Any]:
        """
        Predict mutations for each agent in the pair.
        
        Args:
            agents: List of agent entities [{"start": 27, "end": 32, "text": "Hsp70"}, ...]
            annotated_text: Text with all agents marked with <e></e> tags
            
        Returns:
            Dict with mutations keyed by (start, end, text) tuples
        """
        mutations_dict = {}
        
        for agent in agents:
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
            "agents": agents,
            "annotated_text": annotated_text
        }

    def predict_batch(self, pairs: List[List[Dict]], annotated_texts: List[str]) -> List[Dict[str, Any]]:
        """Predict mutations for multiple pairs using a single batched forward pass.

        We preserve per-agent detection by creating per-agent modified texts,
        running them as a single batch, and regrouping outputs per original item.
        """
        assert len(pairs) == len(annotated_texts)

        # 1) Build per-agent modified texts and an index map back to (item_idx, agent_key)
        modified_texts: List[str] = []
        backrefs: List[Tuple[int, Tuple[int, int, str]]] = []

        for i, (pair_agents, text) in enumerate(zip(pairs, annotated_texts)):
            for agent in pair_agents:
                modified = self._create_single_agent_text(text, agent)
                modified_texts.append(modified)
                backrefs.append((i, (agent["start"], agent["end"], agent["text"])) )

        if not modified_texts:
            return [
                {"mutations": {}, "agents": p, "annotated_text": t}
                for p, t in zip(pairs, annotated_texts)
            ]

        # 2) Tokenize batch
        enc = self.tokenizer(
            modified_texts,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            max_length=512,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offset_mappings = enc["offset_mapping"]  # keep on CPU for indexing

        # 3) Forward pass once
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=2).detach().cpu()

        # 4) Regroup per original item
        grouped_results: List[Dict[str, Any]] = [
            {"mutations": {}, "agents": pairs[i], "annotated_text": annotated_texts[i]}
            for i in range(len(pairs))
        ]

        for idx in range(len(modified_texts)):
            item_idx, agent_key = backrefs[idx]

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[idx].detach().cpu())
            offsets = offset_mappings[idx].tolist()
            predictions = pred_ids[idx].tolist()

            spans = self._extract_mutation_spans(tokens, offsets, predictions, modified_texts[idx])
            if spans:
                muts = self._spans_to_mutations(spans, modified_texts[idx])
                grouped_results[item_idx]["mutations"][agent_key] = muts

        return grouped_results

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
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }
        
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze(0).detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze(0).detach().cpu())
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()
        
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
            List of mutation dictionaries with start, end, text fields (matching training format)
        """
        mutations = []
        
        for span in mutation_spans:
            mutations.append({
                "start": span["start"],
                "end": span["end"], 
                "text": span["text"]
            })
        
        return mutations

