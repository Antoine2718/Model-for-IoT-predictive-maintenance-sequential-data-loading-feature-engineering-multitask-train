import torch
import yaml
from pathlib import Path
from typing import Dict
from .models.digital_twin import DigitalTwin
from .feature_engineering import normalize_batch, add_statistical_channels

class TwinPredictor:
    def __init__(self, cfg_path: str = "./config.yaml"):
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.device = self.cfg["training"]["device"]
        self.ckpt = Path(self.cfg["logging"]["checkpoint_dir"]) / "best.pt"
        # On suppose input_dim connu à l'initialisation via config
        self.input_dim = self.cfg["data"]["num_sensors"] + 1  # + env
        self.model = DigitalTwin(input_dim=self.input_dim, cfg=self.cfg).to(self.device)
        self.model.load_state_dict(torch.load(self.ckpt, map_location=self.device))
        self.model.eval()

    def predict(self, x_np) -> Dict:
        # x_np: [T, F_total]
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        x = normalize_batch(x)
        x = add_statistical_channels(x)
        with torch.no_grad():
            out = self.model(x)
            rul = out["rul"].cpu().numpy().squeeze().item()
            fail_prob = torch.sigmoid(out["fail_logit"]).cpu().numpy().squeeze().item()
            health_seq = out["health_seq"].cpu().numpy().squeeze()
        return {"rul": rul, "failure_probability": fail_prob, "health_sequence": health_seq}
