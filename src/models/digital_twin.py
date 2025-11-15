import torch
import torch.nn as nn
from .sensors_encoder import LSTMEncoder, TransformerEncoder
from ..physics import DegradationModel
from .twin_core import FusionHead

class DigitalTwin(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        enc_type = cfg["model"]["encoder"]["type"]
        if enc_type == "lstm":
            self.encoder = LSTMEncoder(
                input_dim=input_dim,
                hidden_dim=cfg["model"]["encoder"]["d_model"],
                num_layers=2,
                dropout=cfg["model"]["encoder"]["dropout"]
            )
        else:
            self.encoder = TransformerEncoder(
                input_dim=input_dim,
                d_model=cfg["model"]["encoder"]["d_model"],
                n_heads=cfg["model"]["encoder"]["n_heads"],
                n_layers=cfg["model"]["encoder"]["n_layers"],
                dropout=cfg["model"]["encoder"]["dropout"]
            )

        self.physics = DegradationModel(
            init_health=cfg["model"]["twin_core"]["init_health"],
            degradation_rate=cfg["model"]["twin_core"]["degradation_rate"],
            env_weight=cfg["model"]["twin_core"]["env_factor_weight"]
        )
        self.fusion = FusionHead(enc_dim=self.encoder.out_dim, fusion_hidden=cfg["model"]["fusion"]["hidden_dim"])

    def forward(self, x):
        # x: [B, T, F_total] -> derniers canaux supposés: env (1)
        env = x[:, :, -1:].contiguous()
        sensors = x[:, :, :-1].contiguous()
        enc_seq = self.encoder(x)
        health_seq = self.physics(sensors, env)
        rul, fail_logit = self.fusion(enc_seq, health_seq)
        return {
            "rul": rul,               # [B, 1]
            "fail_logit": fail_logit, # [B, 1]
            "health_seq": health_seq  # [B, T, 1]
        }
