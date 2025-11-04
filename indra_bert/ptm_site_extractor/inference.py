from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
from typing import List, Dict, Tuple, Any, Optional


class PTMSiteExtractor:
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

    def predict(self, agent: Dict, annotated_text: str) -> Dict[str, Any]:
        """
        Predict PTM sites for a single agent in annotated text.
        
        Args:
            agent: Agent entity dict with {"start": int, "end": int, "text": str}
            annotated_text: Text with all agents marked with <e></e> tags
            
        Returns:
            Dict with PTM sites: {"sites": [{"residue": "S", "position": 216, ...}], ...}
        """
        # Create modified text with only current agent tagged
        modified_text = self._create_single_agent_text(annotated_text, agent)
        
        # Run PTM site detection on this agent
        site_spans = self._detect_sites_for_agent(modified_text)
        
        # Convert spans to normalized PTM sites
        ptm_sites = []
        if site_spans:
            for span in site_spans:
                normalized = self._normalize_ptm_site(span["text"])
                if normalized:
                    ptm_sites.append({
                        **normalized,
                        "start": span["start"],
                        "end": span["end"],
                        "text": span["text"]
                    })
        
        return {
            "sites": ptm_sites,
            "agent": agent,
            "annotated_text": annotated_text
        }

    def predict_batch(self, agents_list: List[List[Dict]], annotated_texts: List[str]) -> List[Dict[str, Any]]:
        """Predict PTM sites for multiple agents using a single batched forward pass.

        Args:
            agents_list: List of agent lists (one per text)
            annotated_texts: List of annotated texts corresponding to each agent list
            
        Returns:
            List of PTM site prediction results
        """
        assert len(agents_list) == len(annotated_texts)

        # Build per-agent modified texts and backref map
        modified_texts: List[str] = []
        backrefs: List[Tuple[int, Dict]] = []  # (item_idx, agent_dict)

        for i, (agents, text) in enumerate(zip(agents_list, annotated_texts)):
            for agent in agents:
                modified = self._create_single_agent_text(text, agent)
                modified_texts.append(modified)
                backrefs.append((i, agent))

        if not modified_texts:
            return [
                {"sites": [], "agents": agents, "annotated_text": text}
                for agents, text in zip(agents_list, annotated_texts)
            ]

        # Tokenize batch
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
        offset_mappings = enc["offset_mapping"]  # keep on CPU

        # Forward pass once
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=2).detach().cpu()

        # Regroup per original item
        grouped_results: List[Dict[str, Any]] = [
            {"sites": [], "agents": agents_list[i], "annotated_text": annotated_texts[i]}
            for i in range(len(agents_list))
        ]

        for idx in range(len(modified_texts)):
            item_idx, agent = backrefs[idx]

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[idx].detach().cpu())
            offsets = offset_mappings[idx].tolist()
            predictions = pred_ids[idx].tolist()

            spans = self._extract_site_spans(tokens, offsets, predictions, modified_texts[idx])
            for span in spans:
                normalized = self._normalize_ptm_site(span["text"])
                if normalized:
                    grouped_results[item_idx]["sites"].append({
                        **normalized,
                        "start": span["start"],
                        "end": span["end"],
                        "text": span["text"],
                        "agent": agent
                    })

        return grouped_results

    def _create_single_agent_text(self, annotated_text: str, target_agent: Dict) -> str:
        """Create text with only the target agent tagged with <e></e>."""
        # Remove all <e></e> tags first
        clean_text = re.sub(r'<e>(.*?)</e>', r'\1', annotated_text)
        
        # Add back only the target agent's tags
        agent_text = target_agent["text"]
        start = target_agent["start"]
        end = target_agent["end"]
        
        # Insert tags around the target agent
        modified_text = clean_text[:start] + f"<e>{agent_text}</e>" + clean_text[end:]
        
        return modified_text

    def _detect_sites_for_agent(self, text: str) -> List[Dict]:
        """Detect PTM site spans in text for a specific agent."""
        # Tokenize and run model
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            max_length=512,
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
        
        # Extract PTM site spans from predictions
        site_spans = self._extract_site_spans(tokens, offset_mapping, predictions, text)
        
        return site_spans

    def _extract_site_spans(self, tokens: List[str], offset_mapping: List[Tuple], 
                           predictions: List[int], original_text: str) -> List[Dict]:
        """Extract PTM site spans from token-level predictions using BIO tags."""
        site_spans = []
        current_span = None
        
        for i, (token, offset, pred_id) in enumerate(zip(tokens, offset_mapping, predictions)):
            if offset[0] == offset[1]:  # Skip special tokens
                continue
                
            label = self.id2label.get(pred_id, "O")
            
            if label.startswith("B-site"):
                # Start new PTM site span
                if current_span:
                    site_spans.append(current_span)
                current_span = {
                    "start": offset[0],
                    "end": offset[1],
                    "text": original_text[offset[0]:offset[1]]
                }
            elif label.startswith("I-site") and current_span:
                # Continue current PTM site span
                current_span["end"] = offset[1]
                current_span["text"] = original_text[current_span["start"]:offset[1]]
            else:
                # End current span if exists
                if current_span:
                    site_spans.append(current_span)
                    current_span = None
        
        # Add final span if exists
        if current_span:
            site_spans.append(current_span)
        
        return site_spans

    def _normalize_ptm_site(self, site_text: str) -> Optional[Dict[str, Any]]:
        """Normalize PTM site text to residue and position.
        
        Examples:
            "serine-216" -> {"residue": "S", "position": 216}
            "Ser216" -> {"residue": "S", "position": 216}
            "S473" -> {"residue": "S", "position": 473}
            "Thr-187" -> {"residue": "T", "position": 187}
        """
        site_text = site_text.strip()
        
        # Residue name to single letter mapping
        residue_map = {
            "serine": "S", "ser": "S", "s": "S",
            "threonine": "T", "thr": "T", "t": "T",
            "tyrosine": "Y", "tyr": "Y", "y": "Y",
            "lysine": "K", "lys": "K", "k": "K",
            "arginine": "R", "arg": "R", "r": "R",
            "histidine": "H", "his": "H", "h": "H",
            "aspartate": "D", "asp": "D", "d": "D",
            "asparagine": "N", "asn": "N", "n": "N",
            "glutamate": "E", "glu": "E", "e": "E",
            "glutamine": "Q", "gln": "Q", "q": "Q",
            "cysteine": "C", "cys": "C", "c": "C",
            "methionine": "M", "met": "M", "m": "M",
            "phenylalanine": "F", "phe": "F", "f": "F",
            "tryptophan": "W", "trp": "W", "w": "W",
            "leucine": "L", "leu": "L", "l": "L",
            "isoleucine": "I", "ile": "I", "i": "I",
            "valine": "V", "val": "V", "v": "V",
            "proline": "P", "pro": "P", "p": "P",
            "glycine": "G", "gly": "G", "g": "G",
            "alanine": "A", "ala": "A", "a": "A",
        }
        
        # Pattern 1: Single letter + number (e.g., "S473", "T187")
        pattern1 = r'^([A-Z])(\d+)$'
        match = re.match(pattern1, site_text, re.IGNORECASE)
        if match:
            residue = match.group(1).upper()
            position = int(match.group(2))
            if residue in residue_map.values():
                return {"residue": residue, "position": position, "normalized": f"{residue}{position}"}
        
        # Pattern 2: Residue name + separator + number (e.g., "serine-216", "Ser-216", "Thr187")
        pattern2 = r'^([A-Za-z]+)[\s\-]?(\d+)$'
        match = re.match(pattern2, site_text)
        if match:
            residue_name = match.group(1).lower()
            position = int(match.group(2))
            residue = residue_map.get(residue_name)
            if residue:
                return {"residue": residue, "position": position, "normalized": f"{residue}{position}"}
        
        # Pattern 3: Number + residue name (e.g., "216Ser", less common)
        pattern3 = r'^(\d+)[\s\-]?([A-Za-z]+)$'
        match = re.match(pattern3, site_text)
        if match:
            position = int(match.group(1))
            residue_name = match.group(2).lower()
            residue = residue_map.get(residue_name)
            if residue:
                return {"residue": residue, "position": position, "normalized": f"{residue}{position}"}
        
        return None

