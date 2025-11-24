"""
Multi-head token classification model for negation detection.
Two heads: one for cues, one for scopes.
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig


class NegationDetectorModel(PreTrainedModel):
    """
    Model with two token classification heads:
    - cue_head: detects negation cues (B-cue, I-cue, O)
    - scope_head: detects negation scopes (B-scope, I-scope, O)
    """
    
    def __init__(
        self,
        config,
        cue_num_labels: int = 3,  # O, B-cue, I-cue
        scope_num_labels: int = 3,  # O, B-scope, I-scope
    ):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        hidden_size = config.hidden_size
        
        self.cue_num_labels = cue_num_labels
        self.scope_num_labels = scope_num_labels
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cue_classifier = nn.Linear(hidden_size, cue_num_labels)
        self.scope_classifier = nn.Linear(hidden_size, scope_num_labels)
        
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        cue_labels=None,
        scope_labels=None,
        return_dict: bool = True,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        
        sequence_output = self.dropout(outputs.last_hidden_state)
        
        cue_logits = self.cue_classifier(sequence_output)  # [B, L, cue_num_labels]
        scope_logits = self.scope_classifier(sequence_output)  # [B, L, scope_num_labels]
        
        loss = None
        if cue_labels is not None and scope_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            # Cue loss
            active_loss = attention_mask.view(-1) == 1
            active_cue_logits = cue_logits.view(-1, self.cue_num_labels)
            active_cue_labels = torch.where(
                active_loss,
                cue_labels.view(-1),
                torch.tensor(-100).type_as(cue_labels)
            )
            cue_loss = loss_fct(active_cue_logits, active_cue_labels)
            
            # Scope loss
            active_scope_logits = scope_logits.view(-1, self.scope_num_labels)
            active_scope_labels = torch.where(
                active_loss,
                scope_labels.view(-1),
                torch.tensor(-100).type_as(scope_labels)
            )
            scope_loss = loss_fct(active_scope_logits, active_scope_labels)
            
            # Combined loss (equal weight)
            loss = cue_loss + scope_loss
        
        if not return_dict:
            output = (cue_logits, scope_logits)
            return ((loss,) + output) if loss is not None else output
        
        # Store scope_logits in a custom attribute
        from transformers.modeling_outputs import TokenClassifierOutput
        output = TokenClassifierOutput(
            loss=loss,
            logits=cue_logits,  # Return cue_logits as primary logits
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output.scope_logits = scope_logits  # Attach scope_logits
        return output
    
    def get_input_embeddings(self):
        """Get input embeddings from the BERT model."""
        return self.bert.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Set input embeddings for the BERT model."""
        self.bert.set_input_embeddings(value)
    
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings."""
        return self.bert.resize_token_embeddings(new_num_tokens)
    
    def predict(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        """Get predictions for both cues and scopes."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        
        cue_logits = outputs.logits  # [B, L, cue_num_labels]
        scope_logits = outputs.scope_logits  # [B, L, scope_num_labels]
        
        cue_preds = torch.argmax(cue_logits, dim=-1)  # [B, L]
        scope_preds = torch.argmax(scope_logits, dim=-1)  # [B, L]
        
        return {
            "cue_predictions": cue_preds,
            "scope_predictions": scope_preds,
            "cue_logits": cue_logits,
            "scope_logits": scope_logits,
        }

