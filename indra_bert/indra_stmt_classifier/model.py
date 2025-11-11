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
      - gate1: binary relation / no_relation
      - gate2: relation subtype (on relation dataset)
      - gate3: INDRA label (on INDRA benchmark)

    All three heads are always computed.
    Loss uses IGNORE_INDEX masks so different datasets can share the model.
    Trainer only sees:
      - loss (scalar)
      - logits: concatenated [gate1 | gate2 | gate3]
    """

    base_model_prefix = "bert"

    def __init__(
        self,
        config,
        gate2_num_labels: int,
        gate3_num_labels: int,
        gate1_loss_weight: float = 1.0,
        gate2_loss_weight: float = 0.5,
        gate3_loss_weight: float = 0.25,
    ):
        super().__init__(config)

        # backbone encoder
        self.bert = AutoModel.from_config(config)
        hidden_size = config.hidden_size

        # stash for reuse
        config.gate2_num_labels = gate2_num_labels
        config.gate3_num_labels = gate3_num_labels
        config.gate1_loss_weight = gate1_loss_weight
        config.gate2_loss_weight = gate2_loss_weight
        config.gate3_loss_weight = gate3_loss_weight

        self.gate1_loss_weight = gate1_loss_weight
        self.gate2_loss_weight = gate2_loss_weight
        self.gate3_loss_weight = gate3_loss_weight

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # heads
        self.gate1_classifier = nn.Linear(hidden_size, 2)
        self.gate2_classifier = nn.Linear(hidden_size, gate2_num_labels)
        self.gate3_classifier = nn.Linear(hidden_size, gate3_num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        gate1_labels=None,
        gate2_labels=None,
        gate3_labels=None,
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

        # pooled representation
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        pooled = self.dropout(pooled)

        # logits
        gate1_logits = self.gate1_classifier(pooled)  # [B, 2]
        gate2_logits = self.gate2_classifier(pooled)  # [B, N2]
        gate3_logits = self.gate3_classifier(pooled)  # [B, N3]

        combined_logits = torch.cat(
            [gate1_logits, gate2_logits, gate3_logits],
            dim=-1,
        )

        # losses
        loss = None
        gate1_loss = gate2_loss = gate3_loss = None

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

        weighted_losses = []
        if gate1_loss is not None:
            weighted_losses.append(self.gate1_loss_weight * gate1_loss)
        if gate2_loss is not None:
            weighted_losses.append(self.gate2_loss_weight * gate2_loss)
        if gate3_loss is not None:
            weighted_losses.append(self.gate3_loss_weight * gate3_loss)

        if weighted_losses:
            loss = torch.stack(weighted_losses).sum()

        if not return_dict:
            return loss, combined_logits

        return SequenceClassifierOutput(
            loss=loss,
            logits=combined_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=getattr(outputs, "attentions", None),
        )

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

        return {
            "gate1_predictions": gate1_pred,
            "gate1_probs": gate1_probs,
            "gate2_predictions": gate2_pred,
            "gate2_probs": gate2_probs,
            "gate3_predictions": gate3_pred,
            "gate3_probs": gate3_probs,
        }
