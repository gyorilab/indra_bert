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

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])
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
