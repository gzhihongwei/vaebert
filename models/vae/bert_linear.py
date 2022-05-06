import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel

class BertLinearModel(torch.nn.Module):
    def __init__(self, device, latent_dim):
        super().__init__()
        self.device = device
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, 128)
    def forward(self, x):
        tokens = self.bert_tokenizer(x, return_tensors="pt", padding=True).to(self.device)
        bert_outputs = self.bert_model(**tokens)
        cls_tokens = bert_outputs.last_hidden_state[:, 0]
        latent_vector = self.linear(cls_tokens)
        return latent_vector