import torch
import torch.nn as nn


class SmallModel(nn.Module):
    def __init__(self, input_dim=14, emb_dim=32, hidden_dim=64, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, emb_dim)
        enc = nn.TransformerEncoderLayer(emb_dim, 4, hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.regressor = nn.Linear(emb_dim, 1)


    def forward(self, x):
        z = self.embed(x)
        phi = self.encoder(z)
        y = self.regressor(phi.mean(1))
        return y.squeeze(-1), phi