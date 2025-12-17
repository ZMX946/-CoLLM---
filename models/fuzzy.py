import torch
import torch.nn as nn


class FuzzyDecisionAgent(nn.Module):
    def __init__(self, feature_dim, T):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(feature_dim))
        self.sigma = nn.Parameter(torch.ones(feature_dim))
        self.fc = nn.Linear(feature_dim * T, 1)


    def forward(self, phi):
        mu = self.mu.view(1,1,-1)
        sigma = self.sigma.view(1,1,-1) + 1e-6
        M = torch.exp(-((phi - mu) ** 2) / sigma ** 2)
        return torch.sigmoid(self.fc(M.flatten(1))).squeeze(-1)