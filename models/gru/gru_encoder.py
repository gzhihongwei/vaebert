import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, latent_dim):
        dropout_rate = 0.15

        # In place of Keras's SpatialDropout1D: https://github.com/starstorms9/shape/blob/master/archive/text.py#L23
        self.dropout = nn.Dropout2d(dropout_rate)

        self.gru1 = nn.GRU(
            input_size=self.bert.hidden_size,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )
        self.gru2 = nn.GRU(
            input_size=128, hidden_size=128, batch_first=True, bidirectional=True
        )
        self.gru3 = nn.GRU(
            input_size=256, hidden_size=256, batch_first=True, bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=256, out_features=200),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=200, out_features=128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=128, out_features=latent_dim),
        )

    def forward(
        self, bert_embed: torch.Tensor, seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        # https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/2
        bert_embed = bert_embed.permute(0, 2, 1)
        bert_embed = self.dropout(bert_embed)
        bert_embed = bert_embed.permute(0, 2, 1)

        packed_bert_embed = nn.utils.rnn.pack_padded_sequence(
            bert_embed, seq_lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, last_hidden_states = self.gru1(packed_bert_embed)
        packed_output, last_hidden_states = self.gru2(packed_output, last_hidden_states)
        _, last_hidden_states = self.gru3(packed_output, last_hidden_states)
        last_hidden_state = torch.cat(
            (last_hidden_states[-2], last_hidden_states[-1]), dim=1
        )

        output = self.fc(last_hidden_state)

        return output
