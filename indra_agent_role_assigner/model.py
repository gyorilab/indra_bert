from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn.functional as F
import torch
from collections import defaultdict
from pathlib import Path

from indra_agent_role_assigner.preprocess import (
    preprocess_for_inference,
    reassign_member_indices,
    parse_and_generalize_tags,
    map_annotated_to_clean,
    SpecialTokenOffsetFixTokenizer
)


class IndraAgentsTagger:
    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        self.tokenizer = SpecialTokenOffsetFixTokenizer(AutoTokenizer.from_pretrained(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict_tags(self, stmt_type: str, entity_annotated_text: str):
        """
        Returns:
            - BIO tags list
            - confidence scores
            - tokens
            - original offsets (from entity_annotated_text)
            - sequence_ids (to isolate text tokens)
        """
        enc = preprocess_for_inference(stmt_type, entity_annotated_text, self.tokenizer)
        # for tok, offset in zip(enc['tokens'], enc['offsets']):
        #     print(f"{tok}: {offset}")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        
        tokens = enc["tokens"]
        offsets = enc["offsets"]
        seq_ids = enc["sequence_ids"]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probs, dim=2)[0]
            confidence_scores = torch.max(probs, dim=2).values[0]

        pred_labels = [self.id2label[p.item()] if offsets[i] is not None and offsets[i] != (None, None) else "O"
                        for i, p in enumerate(predictions)]
        confidences = [c.item() for c in confidence_scores]

        return pred_labels, confidences, tokens, offsets, seq_ids

    def extract_agents(self, tokens, bio_tags, seq_ids):
        """
        Returns a dictionary of role → list of phrases extracted using BIO tags.
        """
        agents = defaultdict(list)
        current_tag, current_phrase = None, []

        for token, tag, sid in zip(tokens, bio_tags, seq_ids):
            if sid != 1:
                continue
            if tag.startswith("B-"):
                if current_tag and current_phrase:
                    agents[current_tag].append("".join(current_phrase))
                current_tag = tag[2:]
                current_phrase = [token]
            elif tag.startswith("I-") and current_tag:
                current_phrase.append(token)
            else:
                if current_tag and current_phrase:
                    agents[current_tag].append("".join(current_phrase))
                current_tag, current_phrase = None, []

        if current_tag and current_phrase:
            agents[current_tag].append("".join(current_phrase))

        return dict(agents)

    def predict(self, stmt_type: str, annotated_text: str) -> dict:
        """
        Main prediction function.
        Args:
            stmt_type: e.g., "Phosphorylation"
            annotated_text: text with <e>...</e> around each entity
        Returns:
            dict with:
              - clean_text
              - role_spans: clean offsets
              - tokens, bio_tags, confidence scores
              - entity-annotated and role-annotated versions of the text
        """
        # Step 1: Convert <role> to <e> and get clean text
        clean_text, _, _ = parse_and_generalize_tags(annotated_text)

        # Step 2: Predict using entity-annotated text
        bio_tags, confidences, tokens, offsets, seq_ids = self.predict_tags(stmt_type, annotated_text)

        # Step 3: Normalize member labels to members.0, .1, etc.
        bio_tags = reassign_member_indices(bio_tags)

        # Step 4: Replace None offsets with "O" in BIO tags
        bio_tags = [
            "O" if offset is None or offset == (None, None) else tag
            for tag, offset in zip(bio_tags, offsets)
        ]

        # Step 5: Map offsets from annotated_text to clean_text
        offset_map = map_annotated_to_clean(annotated_text, clean_text)
        role_spans = []
        open_role = None
        role_start = None
        prev_end = None

        for i, (label, offset, sid) in enumerate(zip(bio_tags, offsets, seq_ids)):
            if sid != 1 or offset is None or offset == (0, 0):
                continue

            start_char, end_char = offset
            if start_char is None or end_char is None:
                continue  # skip special tokens

            clean_start = offset_map.get(start_char)
            clean_end = offset_map.get(end_char - 1)
            if clean_start is None or clean_end is None:
                continue
            clean_end += 1  # end is inclusive

            if label.startswith("B-"):
                if open_role is not None:
                    role_spans.append({
                        "role": open_role,
                        "start": role_start,
                        "end": prev_end,
                        "text": clean_text[role_start:prev_end]
                    })
                open_role = label[2:]
                role_start = clean_start
            elif label.startswith("I-") and open_role:
                pass
            else:
                if open_role is not None:
                    role_spans.append({
                        "role": open_role,
                        "start": role_start,
                        "end": prev_end,
                        "text": clean_text[role_start:prev_end]
                    })
                    open_role = None
                    role_start = None

            prev_end = clean_end

        if open_role and role_start is not None:
            role_spans.append({
                "role": open_role,
                "start": role_start,
                "end": prev_end,
                "text": clean_text[role_start:prev_end]
            })

        # Step 5: Annotate clean text with <role> tags
        agents = self.extract_agents(tokens, bio_tags, seq_ids)

        return {
            "stmt_type": stmt_type,
            "text": annotated_text,
            "clean_text": clean_text,
            "tokens": tokens,
            "bio_tags": bio_tags,
            "confidence": confidences,
            "agents": agents,
            "role_spans": role_spans
        }
