import torch
import torch.nn as nn

from transformers import BertModel


class BERTEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        dropout_rate = 0.15

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(self.bert.config.hidden_size, latent_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_tokens = self.dropout(bert_outputs.last_hidden_state[:, 0])
        output = self.linear(cls_tokens)
        return output
