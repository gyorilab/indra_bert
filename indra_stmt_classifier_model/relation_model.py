import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForIndraStmtClassification(PreTrainedModel):
    config_class = AutoConfig  # This tells HF to use AutoConfig when loading/saving configs

    def __init__(self, config):
        super().__init__(config)

        # Load the base model (BERT or other transformer)
        self.bert = AutoModel.from_config(config)

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights (this initializes classifier weights properly)
        self.post_init()

    @classmethod
    def from_pretrained_with_labels(cls, pretrained_model_name, label2id, id2label, **kwargs):
        # Load config from pretrained and update label mappings
        config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
            **kwargs
        )
        # Initialize model
        model = cls(config)
        # Load pretrained weights
        model.bert = AutoModel.from_pretrained(pretrained_model_name, config=config)
        return model

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get encoder outputs
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )

        # Get CLS token embedding
        pooled_output = outputs.last_hidden_state[:, 0]

        # Classification head
        logits = self.classifier(pooled_output)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        # Return standard HuggingFace classification output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
