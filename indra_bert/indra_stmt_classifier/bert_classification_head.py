import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PreTrainedModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from time import time


class TwoGatedClassifier(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Gate 1: Binary classification (relation vs no_relation)
        self.gate1_classifier = nn.Linear(config.hidden_size, 2)
        
        # Gate 2: Multi-class classification (specific relation types)
        self.gate2_classifier = nn.Linear(config.hidden_size, config.num_labels)
        
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
            id2label=id2label
        )
        
        # Manually assign any extra fields
        for key, value in kwargs.items():
            setattr(config, key, value)

        model = cls(config)
        model.bert = AutoModel.from_pretrained(pretrained_model_name, config=config)
        return model

    def forward(self, 
                input_ids, 
                attention_mask=None, 
                token_type_ids=None, 
                labels=None, 
                gate1_labels=None, 
                class_weights=None, 
                gate1_weights=None
        ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        last_hidden = outputs.last_hidden_state  # [B, T, H]
        cls_embeddings = last_hidden[:, 0, :]  # [B, H]
        pooled_output = self.dropout(cls_embeddings)
        
        # Gate 1: Binary classification
        gate1_logits = self.gate1_classifier(pooled_output)
        
        # Gate 2: Multi-class classification
        gate2_logits = self.gate2_classifier(pooled_output)

        loss = None
        if labels is not None and gate1_labels is not None:
            # Gate 1 loss (binary) - CrossEntropy with precomputed class weights
            if not isinstance(gate1_weights, torch.Tensor):
                gate1_weights = torch.tensor(gate1_weights, device=gate1_logits.device, dtype=gate1_logits.dtype)
            gate1_loss_fn = nn.CrossEntropyLoss(weight=gate1_weights)
            gate1_loss = gate1_loss_fn(gate1_logits, gate1_labels)
            
            # Gate 2 loss (only for positive examples) - CrossEntropy with class weights
            positive_mask = (gate1_labels == 1)
            if positive_mask.sum() > 0:
                # Standard CrossEntropy loss with class weights for rare relations
                weight_tensor = torch.tensor(class_weights, device=gate2_logits.device, dtype=gate2_logits.dtype) if class_weights is not None else None
                gate2_loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
                gate2_loss = gate2_loss_fn(gate2_logits[positive_mask], labels[positive_mask])
            else:
                gate2_loss = torch.tensor(0.0, device=gate1_logits.device)
            
            # Combined loss
            loss = gate1_loss + gate2_loss

        # Build result dictionary
        if labels is not None and gate1_labels is not None:
            # Training mode - return tuple format for compute_metrics
            result_dict = {
                'loss': loss,
                'logits': (gate1_logits, gate2_logits),  # Tuple: (gate1, gate2) for compute_metrics
            }
        else:
            # Inference mode - return separate logits for predict method
            result_dict = {
                'gate1_logits': gate1_logits,
                'gate2_logits': gate2_logits,
            }
        return result_dict


    def predict(self, input_ids, attention_mask=None, token_type_ids=None):
        """Two-gated inference with learnable threshold"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, token_type_ids)
            
            # Gate 1: Binary classification (relation vs no_relation)
            gate1_probs = torch.softmax(outputs['gate1_logits'], dim=-1)
            has_relation_prob = gate1_probs[:, 1]  # Probability of having a relation
            
            # Gate 2: Multi-class classification (relation types)
            gate2_probs = torch.softmax(outputs['gate2_logits'], dim=-1)
            
            # Decision logic using argmax (standard approach)
            predictions = []
            confidences = []
            
            for i in range(len(has_relation_prob)):
                gate1_decision = torch.argmax(gate1_probs[i])  # 0=no_relation, 1=has_relation
                
                if gate1_decision == 1:
                    # Gate 1 says "has relation" → use Gate 2 to classify type
                    relation_type_idx = torch.argmax(gate2_probs[i])
                    confidence = has_relation_prob[i] * gate2_probs[i][relation_type_idx]
                    predictions.append(relation_type_idx.item())
                    confidences.append(confidence.item())
                else:
                    # Gate 1 says "no relation" → return -1
                    predictions.append(-1)
                    confidences.append(gate1_probs[i][0].item())  # Confidence in no_relation
            
            return {
                'predictions': predictions,
                'confidences': confidences,
                'gate1_probs': gate1_probs,
                'gate2_probs': gate2_probs
            }
