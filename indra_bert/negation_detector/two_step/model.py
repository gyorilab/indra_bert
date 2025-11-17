"""
Two-stage negation detection model:
1. Cue detection: token classification (BIO)
2. Scope prediction: span-based (start/end positions given a cue)
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig


class NegationDetectorModel(PreTrainedModel):
    """
    Two-stage model:
    - cue_head: detects negation cues (B-cue, I-cue, O) - token classification
    - scope_head: predicts scope start/end positions given a cue - span prediction
    """
    
    def __init__(
        self,
        config,
        cue_num_labels: int = 3,  # O, B-cue, I-cue
    ):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        hidden_size = config.hidden_size
        
        self.cue_num_labels = cue_num_labels
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Cue detection head (token classification)
        self.cue_classifier = nn.Linear(hidden_size, cue_num_labels)
        
        # Scope prediction head (span extraction: start and end positions)
        # Given a cue position, predict scope start and end
        # We'll add cue position information to help the model
        self.scope_start_classifier = nn.Linear(hidden_size, 1)  # Start position logits
        self.scope_end_classifier = nn.Linear(hidden_size, 1)    # End position logits
        
        # Optional: cue position embedding to condition scope prediction on cue
        # This helps the model know which cue we're predicting scope for
        self.cue_position_embedding = nn.Embedding(512, hidden_size)  # Max seq len
        
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        cue_labels=None,
        scope_start_positions=None,  # [B] - start position for scope (token index)
        scope_end_positions=None,    # [B] - end position for scope (token index)
        cue_positions=None,           # [B] - position of the cue (token index) for scope prediction
        return_dict: bool = True,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        
        sequence_output = self.dropout(outputs.last_hidden_state)  # [B, L, H]
        
        # Cue detection logits
        cue_logits = self.cue_classifier(sequence_output)  # [B, L, cue_num_labels]
        
        # Scope prediction: condition on cue position if provided
        if cue_positions is not None:
            # Add cue position embedding to sequence output
            batch_size, seq_len, hidden_size = sequence_output.shape
            cue_pos_emb = self.cue_position_embedding(cue_positions)  # [B, H]
            cue_pos_emb = cue_pos_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, H]
            scope_input = sequence_output + cue_pos_emb  # Add cue position signal
        else:
            scope_input = sequence_output
        
        # Scope prediction logits (start and end)
        scope_start_logits = self.scope_start_classifier(scope_input).squeeze(-1)  # [B, L]
        scope_end_logits = self.scope_end_classifier(scope_input).squeeze(-1)      # [B, L]
        
        loss = None
        if cue_labels is not None:
            # Cue classification loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_cue_logits = cue_logits.view(-1, self.cue_num_labels)
            active_cue_labels = torch.where(
                active_loss,
                cue_labels.view(-1),
                torch.tensor(-100).type_as(cue_labels)
            )
            cue_loss = loss_fct(active_cue_logits, active_cue_labels)
            loss = cue_loss
        
        if scope_start_positions is not None and scope_end_positions is not None:
            # Scope span prediction loss
            # We predict start/end positions for the scope given a cue
            # Use cross-entropy over sequence positions
            
            # Mask out invalid positions
            batch_size, seq_len = scope_start_logits.shape
            position_mask = attention_mask.bool()  # [B, L]
            
            # Start position loss
            scope_start_loss = self._span_loss(
                scope_start_logits, scope_start_positions, position_mask
            )
            
            # End position loss
            scope_end_loss = self._span_loss(
                scope_end_logits, scope_end_positions, position_mask
            )
            
            scope_loss = scope_start_loss + scope_end_loss
            
            if loss is not None:
                loss = loss + scope_loss
            else:
                loss = scope_loss
        
        if not return_dict:
            output = (cue_logits, scope_start_logits, scope_end_logits)
            return ((loss,) + output) if loss is not None else output
        
        from transformers.modeling_outputs import TokenClassifierOutput
        output = TokenClassifierOutput(
            loss=loss,
            logits=cue_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output.scope_start_logits = scope_start_logits
        output.scope_end_logits = scope_end_logits
        return output
    
    def _span_loss(self, logits, target_positions, mask):
        """
        Compute loss for span position prediction.
        
        Args:
            logits: [B, L] logits for each position
            target_positions: [B] target position indices
            mask: [B, L] boolean mask for valid positions
        """
        # Convert to probability distribution over positions
        # Mask out invalid positions
        masked_logits = logits.masked_fill(~mask, float('-inf'))
        probs = torch.softmax(masked_logits, dim=-1)  # [B, L]
        
        # Create one-hot targets
        batch_size, seq_len = logits.shape
        targets = torch.zeros(batch_size, seq_len, device=logits.device)
        for i, pos in enumerate(target_positions):
            if 0 <= pos < seq_len:
                targets[i, pos] = 1.0
        
        # Cross-entropy loss
        loss = -torch.sum(targets * torch.log(probs + 1e-8), dim=-1)
        return loss.mean()
    
    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.bert.set_input_embeddings(value)
    
    def resize_token_embeddings(self, new_num_tokens):
        return self.bert.resize_token_embeddings(new_num_tokens)
    
    def predict_cues(self, input_ids, attention_mask=None, token_type_ids=None):
        """Predict negation cues."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        
        cue_logits = outputs.logits  # [B, L, cue_num_labels]
        cue_preds = torch.argmax(cue_logits, dim=-1)  # [B, L]
        
        return cue_preds, cue_logits
    
    def predict_scope(self, input_ids, attention_mask=None, token_type_ids=None, cue_position=None):
        """
        Predict scope span given a cue position.
        
        Args:
            input_ids: [B, L] input token IDs
            attention_mask: [B, L] attention mask
            token_type_ids: [B, L] token type IDs
            cue_position: [B] cue position (token index) - if None, uses first B-cue token
        
        Returns:
            scope_start_preds: [B] predicted start positions
            scope_end_preds: [B] predicted end positions
            scope_start_logits: [B, L] start position logits
            scope_end_logits: [B, L] end position logits
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        
        scope_start_logits = outputs.scope_start_logits  # [B, L]
        scope_end_logits = outputs.scope_end_logits      # [B, L]
        
        # Mask invalid positions
        masked_start_logits = scope_start_logits.masked_fill(~attention_mask.bool(), float('-inf'))
        masked_end_logits = scope_end_logits.masked_fill(~attention_mask.bool(), float('-inf'))
        
        # Predict positions
        scope_start_preds = torch.argmax(masked_start_logits, dim=-1)  # [B]
        scope_end_preds = torch.argmax(masked_end_logits, dim=-1)      # [B]
        
        return scope_start_preds, scope_end_preds, scope_start_logits, scope_end_logits

