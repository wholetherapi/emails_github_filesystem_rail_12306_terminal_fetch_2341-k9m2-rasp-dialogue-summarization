"""Retrieval-augmented sentence planner."""

import torch
import torch.nn as nn


class SentencePlanner(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.plan_head = nn.Linear(backbone.config.d_model, 32)
        self.summary_head = nn.Linear(backbone.config.d_model, backbone.config.vocab_size)

    def forward(self, input_ids, attention_mask, retrieved_ids):
        encoded = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoded.last_hidden_state.mean(dim=1)
        retrieved = self.backbone(input_ids=retrieved_ids).last_hidden_state.mean(dim=1)
        fused = hidden + 0.3 * retrieved
        plan = torch.softmax(self.plan_head(fused), dim=-1)
        tokens = self.summary_head(fused)
        return tokens, plan
