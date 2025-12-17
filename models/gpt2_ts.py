import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class GPT2TimeSeries(nn.Module):
    def __init__(self, input_dim=14, patch_size=4):
        super().__init__()
        self.patch = patch_size

        # ✅ 完全离线，不联网
        config = GPT2Config()
        self.gpt = GPT2Model(config)

        # 冻结 GPT-2
        for p in self.gpt.parameters():
            p.requires_grad = False

        h = self.gpt.config.hidden_size
        self.proj = nn.Linear(input_dim * patch_size, h)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        B, T, d = x.shape
        patches = []
        for i in range(0, T - self.patch + 1, self.patch):
            patches.append(x[:, i:i + self.patch, :].reshape(B, -1))
        patches = torch.stack(patches, 1)

        emb = self.proj(patches)
        out = self.gpt(inputs_embeds=emb).last_hidden_state
        y = self.head(out.mean(1))
        return y.squeeze(-1), out
