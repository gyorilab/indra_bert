import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class BertForIndraStmtClassification(PreTrainedModel):
    config_class = AutoConfig  # Or dynamic if supporting multiple models

    def __init__(self, config):
        super().__init__(config)

        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.entity_proj = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(3 * config.hidden_size, config.num_labels)

        self.post_init()

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.bert.set_input_embeddings(new_embeddings)

    @classmethod
    def from_pretrained_with_labels(cls, pretrained_model_name, label2id, id2label, **kwargs):
        config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
            **kwargs
        )
        config.pretrained_model_name_or_path = pretrained_model_name
        model = cls(config)
        model.bert = AutoModel.from_pretrained(pretrained_model_name, config=config)
        return model

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None,
                entity_token_spans=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        last_hidden = outputs.last_hidden_state  # [B, T, H]
        batch_size = input_ids.size(0)
        hidden_size = last_hidden.size(-1)

        cls_embeddings = last_hidden[:, 0, :]  # [B, H]
        entity_repr = last_hidden.new_zeros(batch_size, hidden_size)

        for i in range(batch_size):
            spans = entity_token_spans[i]  # list of (start, end)
            span_embeddings = [
                torch.cat((last_hidden[i, start], last_hidden[i, end - 1]), dim=-1)
                for start, end in spans
            ]
            if span_embeddings:
                span_stack = torch.stack(span_embeddings, dim=0)
                span_mean = span_stack.mean(dim=0)
                span_max = span_stack.max(dim=0).values
                entity_repr[i] = self.entity_proj(torch.cat([span_mean, span_max], dim=-1))  

        abs_diff = torch.abs(entity_repr - cls_embeddings)  # [B, H]
        relation_repr = torch.cat([
            cls_embeddings, entity_repr, abs_diff
        ], dim=-1)

        pooled_output = self.dropout(relation_repr)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
