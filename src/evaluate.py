import torch
import yaml
from pathlib import Path
from .data_loader import get_dataloaders
from .feature_engineering import normalize_batch, add_statistical_channels
from .models.digital_twin import DigitalTwin
from .losses import Metrics

def evaluate(cfg_path: str = "./config.yaml"):
    train_loader, val_loader, test_loader, cfg = get_dataloaders(cfg_path)
    input_dim = next(iter(train_loader))["x"].shape[-1]
    model = DigitalTwin(input_dim=input_dim, cfg=cfg).to(cfg["training"]["device"])

    ckpt = Path(cfg["logging"]["checkpoint_dir"]) / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=cfg["training"]["device"]))
    model.eval()

    metrics = Metrics()
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(cfg["training"]["device"])
            x = normalize_batch(x)
            x = add_statistical_channels(x)
            y_rul = batch["y_rul"].to(cfg["training"]["device"])
            y_fail = batch["y_fail"].to(cfg["training"]["device"])
            preds = model(x)
            metrics.update(preds, {"y_rul": y_rul, "y_fail": y_fail})
    stats = metrics.compute()
    print(f"Test MSE RUL: {stats['mse_rul']:.4f}, Test AUROC Failure: {stats['auroc_fail']:.4f}, Test Acc Failure: {stats['acc_fail']:.4f}")
    return stats

if __name__ == "__main__":
    evaluate()
