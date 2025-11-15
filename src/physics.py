import torch
import torch.nn as nn

class DegradationModel(nn.Module):
    """
    Modèle physique simplifié:
    h_{t+1} = h_t - (alpha * load_t + beta * env_t) * dt
    Contraintes: h in [0,1]
    """
    def __init__(self, init_health=1.0, degradation_rate=1e-4, env_weight=0.3):
        super().__init__()
        self.init_health = init_health
        self.alpha = nn.Parameter(torch.tensor(degradation_rate))
        self.beta = nn.Parameter(torch.tensor(env_weight))

    def forward(self, sensor_feats: torch.Tensor, env: torch.Tensor):
        # sensor_feats: [B, T, F_s], env: [B, T, 1]
        # On approxime load_t par norme des capteurs (vibration/temperature synthétiques)
        load = sensor_feats.norm(dim=-1, keepdim=True)
        dt = 1.0
        B, T, _ = load.shape
        h = torch.full((B, 1, 1), self.init_health, device=load.device)
        h_hist = []
        for t in range(T):
            dh = (self.alpha * load[:, t:t+1, :] + self.beta * env[:, t:t+1, :]) * dt
            h = torch.clamp(h - dh, min=0.0, max=1.0)
            h_hist.append(h)
        h_seq = torch.cat(h_hist, dim=1)  # [B, T, 1]
        return h_seq
