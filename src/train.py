"""Training and evaluation routines."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data import build_dataloaders, load_spectra_and_concentrations
from src.model import CNNRegressor
from src.utils import get_device, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        loss = criterion(model(inputs), targets)
        total_loss += loss.item() * targets.size(0)
        total_n += targets.size(0)
    return total_loss / total_n


def train(cfg: dict):
    """Full training run driven by a config dict (loaded from config.yaml)."""
    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])

    # Data
    spectra, concentrations = load_spectra_and_concentrations(
        cfg["data"]["spectra_csv"],
        cfg["data"]["concentration_csv"],
    )
    loaders = build_dataloaders(
        spectra,
        concentrations,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        batch_size=cfg["training"]["batch_size"],
        device=str(device),
    )

    # Model
    mcfg = cfg["model"]
    model = CNNRegressor(
        in_channels=mcfg["in_channels"],
        conv1_out=mcfg["conv1_out"],
        conv2_out=mcfg["conv2_out"],
        kernel_size=mcfg["kernel_size"],
        pool_size=mcfg["pool_size"],
        fc1_out=mcfg["fc1_out"],
        num_targets=mcfg["num_targets"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    # Checkpoint directory
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss = evaluate(model, loaders["val"], criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

    # Test
    test_loss = evaluate(model, loaders["test"], criterion, device)
    print(f"\nTest loss: {test_loss:.6f}")
    history["test_loss"] = test_loss

    # Save artifacts
    torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Loss history saved to {log_dir / 'history.json'}")

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
