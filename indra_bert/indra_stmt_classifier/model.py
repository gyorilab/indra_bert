import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

IGNORE_INDEX = -100


class MultiHeadStmtClassifier(PreTrainedModel):
    """
    Multi-head statement classifier with:
      - gate1: binary relation / no_relation (sequence classification)
      - gate2: relation subtype (on relation dataset, sequence classification)
      - gate3: INDRA label (on INDRA benchmark, sequence classification)
      - gate4: predicate detection (HRT dataset, token classification)

    All four heads are always computed.
    Loss uses IGNORE_INDEX masks so different datasets can share the model.
    Trainer only sees:
      - loss (scalar)
      - logits: concatenated [gate1 | gate2 | gate3] for sequence classification
      - gate4_logits: [B, L, num_labels] for token classification
    """

    base_model_prefix = "bert"

    def __init__(
        self,
        config,
        gate2_num_labels: int,
        gate3_num_labels: int,
        gate4_num_labels: int = 3,  # O, B-trigger, I-trigger
        gate1_loss_weight: float = 1.0,
        gate2_loss_weight: float = 0.5,
        gate3_loss_weight: float = 0.25,
        gate4_loss_weight: float = 0.5,
    ):
        super().__init__(config)

        # backbone encoder
        self.bert = AutoModel.from_config(config)
        hidden_size = config.hidden_size

        # stash for reuse
        config.gate2_num_labels = gate2_num_labels
        config.gate3_num_labels = gate3_num_labels
        config.gate4_num_labels = gate4_num_labels
        config.gate1_loss_weight = gate1_loss_weight
        config.gate2_loss_weight = gate2_loss_weight
        config.gate3_loss_weight = gate3_loss_weight
        config.gate4_loss_weight = gate4_loss_weight

        self.gate1_loss_weight = gate1_loss_weight
        self.gate2_loss_weight = gate2_loss_weight
        self.gate3_loss_weight = gate3_loss_weight
        self.gate4_loss_weight = gate4_loss_weight

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # sequence classification heads
        self.gate1_classifier = nn.Linear(hidden_size, 2)
        self.gate2_classifier = nn.Linear(hidden_size, gate2_num_labels)
        self.gate3_classifier = nn.Linear(hidden_size, gate3_num_labels)
        
        # token classification head for predicate detection
        self.gate4_classifier = nn.Linear(hidden_size, gate4_num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        gate1_labels=None,
        gate2_labels=None,
        gate3_labels=None,
        gate4_labels=None,  # token-level labels [B, L]
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # pooled representation for sequence classification
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        pooled = self.dropout(pooled)

        # sequence classification logits
        gate1_logits = self.gate1_classifier(pooled)  # [B, 2]
        gate2_logits = self.gate2_classifier(pooled)  # [B, N2]
        gate3_logits = self.gate3_classifier(pooled)  # [B, N3]

        combined_logits = torch.cat(
            [gate1_logits, gate2_logits, gate3_logits],
            dim=-1,
        )

        # token classification logits for predicate detection
        sequence_output = self.dropout(outputs.last_hidden_state)
        gate4_logits = self.gate4_classifier(sequence_output)  # [B, L, num_labels]

        # losses
        loss = None
        gate1_loss = gate2_loss = gate3_loss = gate4_loss = None

        if gate1_labels is not None:
            mask1 = gate1_labels != IGNORE_INDEX
            if mask1.any():
                gate1_loss = F.cross_entropy(
                    gate1_logits[mask1],
                    gate1_labels[mask1],
                )

        if gate2_labels is not None:
            mask2 = gate2_labels != IGNORE_INDEX
            if mask2.any():
                gate2_loss = F.cross_entropy(
                    gate2_logits[mask2],
                    gate2_labels[mask2],
                )

        if gate3_labels is not None:
            mask3 = gate3_labels != IGNORE_INDEX
            if mask3.any():
                gate3_loss = F.cross_entropy(
                    gate3_logits[mask3],
                    gate3_labels[mask3],
                )

        if gate4_labels is not None:
            # Token classification loss
            # gate4_labels: [B, L], gate4_logits: [B, L, num_labels]
            active_loss = attention_mask.view(-1) == 1
            active_logits = gate4_logits.view(-1, self.config.gate4_num_labels)
            active_labels = torch.where(
                active_loss,
                gate4_labels.view(-1),
                torch.tensor(IGNORE_INDEX).type_as(gate4_labels)
            )
            gate4_loss = F.cross_entropy(
                active_logits,
                active_labels,
                ignore_index=IGNORE_INDEX,
            )

        weighted_losses = []
        if gate1_loss is not None:
            weighted_losses.append(self.gate1_loss_weight * gate1_loss)
        if gate2_loss is not None:
            weighted_losses.append(self.gate2_loss_weight * gate2_loss)
        if gate3_loss is not None:
            weighted_losses.append(self.gate3_loss_weight * gate3_loss)
        if gate4_loss is not None:
            weighted_losses.append(self.gate4_loss_weight * gate4_loss)

        if weighted_losses:
            loss = torch.stack(weighted_losses).sum()

        if not return_dict:
            return loss, combined_logits, gate4_logits

        # Custom output to include gate4_logits
        output = SequenceClassifierOutput(
            loss=loss,
            logits=combined_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=getattr(outputs, "attentions", None),
        )
        output.gate4_logits = gate4_logits
        return output

    @torch.no_grad()
    def predict(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        threshold: float = 0.5,
    ):
        self.eval()
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        logits = outputs.logits

        gate2_num = self.config.gate2_num_labels
        gate3_num = self.config.gate3_num_labels

        gate1_logits = logits[:, :2]
        gate2_logits = logits[:, 2 : 2 + gate2_num]
        gate3_logits = logits[:, 2 + gate2_num : 2 + gate2_num + gate3_num]

        gate1_probs = torch.softmax(gate1_logits, dim=-1)
        gate2_probs = torch.softmax(gate2_logits, dim=-1)
        gate3_probs = torch.softmax(gate3_logits, dim=-1)

        has_relation_prob = gate1_probs[:, 1]
        gate1_pred = (has_relation_prob > threshold).long()
        gate2_pred = torch.argmax(gate2_probs, dim=-1)
        gate3_pred = torch.argmax(gate3_probs, dim=-1)

        # Gate4: token classification predictions
        gate4_logits = outputs.gate4_logits  # [B, L, num_labels]
        gate4_pred = torch.argmax(gate4_logits, dim=-1)  # [B, L]

        return {
            "gate1_predictions": gate1_pred,
            "gate1_probs": gate1_probs,
            "gate2_predictions": gate2_pred,
            "gate2_probs": gate2_probs,
            "gate3_predictions": gate3_pred,
            "gate3_probs": gate3_probs,
            "gate4_predictions": gate4_pred,  # [B, L]
            "gate4_logits": gate4_logits,  # [B, L, num_labels]
        }
