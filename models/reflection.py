import torch
import torch.nn as nn


class SelfReflection(nn.Module):
    def __init__(self, feature_dim, T):
        super().__init__()
        self.fc = nn.Linear(feature_dim * T, 1)
    

    def forward(self, phi):
        return torch.sigmoid(self.fc(phi.flatten(1))).squeeze(-1)