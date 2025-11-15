import torch
import torch.nn as nn

class FusionHead(nn.Module):
    def __init__(self, enc_dim, fusion_hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim + 1, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
        )
        self.rul_head = nn.Sequential(
            nn.Linear(fusion_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fail_head = nn.Sequential(
            nn.Linear(fusion_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, enc_seq, health_seq):
        # enc_seq: [B, T, H], health_seq: [B, T, 1]
        fused = torch.cat([enc_seq, health_seq], dim=-1)
        # On prend la dernière étape temporelle pour prédiction RUL/failure instantanée
        last = fused[:, -1, :]
        z = self.mlp(last)
        rul = self.rul_head(z)
        fail_logit = self.fail_head(z)
        return rul, fail_logit
