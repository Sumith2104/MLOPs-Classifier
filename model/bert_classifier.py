"""
bert_classifier.py
-------------------
Dual-head BERT model for simultaneous Intent + Priority classification.

Architecture:
  - Backbone : bert-base-uncased (frozen or fine-tuned)
  - Head 1   : Linear(hidden_size → num_intent_labels)
  - Head 2   : Linear(hidden_size → num_priority_labels)
  - Both heads share the same [CLS] token representation.

Usage:
    from model.bert_classifier import DualHeadBertClassifier
    model = DualHeadBertClassifier(num_intent_labels=3, num_priority_labels=3)
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class DualHeadBertClassifier(nn.Module):
    """
    BertModel backbone with two independent classification heads:
      - intent_head   : classifies into complaint / inquiry / feedback
      - priority_head : classifies into high / medium / low
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_intent_labels: int = 3,
        num_priority_labels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        self.dropout = nn.Dropout(dropout)

        # Classification heads
        self.intent_head   = nn.Linear(hidden_size, num_intent_labels)
        self.priority_head = nn.Linear(hidden_size, num_priority_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                'intent_logits'   – shape (batch, num_intent_labels)
                'priority_logits' – shape (batch, num_priority_labels)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # [CLS] token representation
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])

        return {
            "intent_logits":   self.intent_head(cls_output),
            "priority_logits": self.priority_head(cls_output),
        }

    def compute_loss(
        self,
        intent_logits: torch.Tensor,
        priority_logits: torch.Tensor,
        intent_labels: torch.Tensor,
        priority_labels: torch.Tensor,
        intent_weight: float = 1.0,
        priority_weight: float = 1.0,
    ) -> torch.Tensor:
        """Combined cross-entropy loss across both heads."""
        ce_loss = nn.CrossEntropyLoss()
        intent_loss   = ce_loss(intent_logits,   intent_labels)
        priority_loss = ce_loss(priority_logits, priority_labels)
        return intent_weight * intent_loss + priority_weight * priority_loss
