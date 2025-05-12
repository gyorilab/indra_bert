from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pathlib import Path

from indra_stmt_agents_ner_model.preprocess import preprocess_for_inference, reassign_member_indices


class IndraAgentsTagger:
    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict_tags(self, stmt_type: str, text: str):
        """
        Returns:
            - BIO tags list (e.g., ["O", "B-subj", "I-subj", ...])
            - tokens (for debugging/visualization)
            - offsets (for annotation)
            - sequence_ids (to isolate text tokens)
        """
        enc = preprocess_for_inference(stmt_type, text, self.tokenizer)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        tokens = enc["tokens"]
        offsets = enc["offsets"]
        seq_ids = enc["sequence_ids"]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        predictions = torch.argmax(outputs.logits, dim=2)[0]
        pred_labels = [self.id2label[p.item()] for p in predictions]

        return pred_labels, tokens, offsets, seq_ids

    def annotate_text(self, text: str, offsets, sequence_ids, bio_tags):
        """
        Given text, offsets, and BIO tags, return annotated text with inline <role>...</role> tags.
        """
        annotated_text = ""
        last_end = 0
        open_tag = None

        for idx, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
            start, end = offset
            if seq_id != 1 or start is None or start == end:
                continue

            word = text[start:end]
            if start > last_end:
                annotated_text += text[last_end:start]

            label = bio_tags[idx]
            if label.startswith("B-"):
                if open_tag:
                    annotated_text += f"</{open_tag}>"
                open_tag = label[2:]
                annotated_text += f"<{open_tag}>{word}"
            elif label.startswith("I-") and open_tag:
                annotated_text += word
            else:
                if open_tag:
                    annotated_text += f"</{open_tag}>"
                    open_tag = None
                annotated_text += word

            last_end = end

        if open_tag:
            annotated_text += f"</{open_tag}>"

        return annotated_text

    def predict(self, stmt_type: str, text: str) -> dict:
        """
        End-to-end tag prediction and annotation.
        """
        bio_tags, tokens, offsets, seq_ids = self.predict_tags(stmt_type, text)

         # Reassign members.0, .1, ... dynamically
        bio_tags = reassign_member_indices(bio_tags)

        annotated = self.annotate_text(text, offsets, seq_ids, bio_tags)

        return {
            "annotated_text": annotated,
            "tokens": tokens,
            "bio_tags": bio_tags
        }
