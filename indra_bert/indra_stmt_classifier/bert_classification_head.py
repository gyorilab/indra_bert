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
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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

        last_hidden = outputs.last_hidden_state  # shape: [B, T, H]
        batch_size = input_ids.size(0)
        hidden_size = last_hidden.size(-1)

        entity_repr = torch.zeros(batch_size, hidden_size).to(last_hidden.device)
        for i in range(batch_size):
            spans = entity_token_spans[i]  # list of [start, end]
            span_embeddings = [last_hidden[i, start:end].mean(dim=0) for start, end in spans if end > start]
            if span_embeddings:
                entity_repr[i] = torch.stack(span_embeddings, dim=0).mean(dim=0)

        pooled_output = self.dropout(entity_repr)  # shape: [B, H]
        logits = self.classifier(pooled_output)
        ### --- END DIFF ---

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
