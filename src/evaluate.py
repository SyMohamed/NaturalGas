"""Generate plots from a trained model: loss curves, pred-vs-actual, R² per species."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import r2_score

from src.data import build_dataloaders, load_spectra_and_concentrations
from src.model import CNNRegressor
from src.utils import get_device


# ── Helpers ──────────────────────────────────────────────────────

SPECIES = ["CH4", "C2H6", "C3H8", "nC4H10", "iC4H10", "nC5H12", "iC5H12"]


@torch.no_grad()
def predict_all(model, loader, device):
    """Run model on an entire DataLoader, return (predictions, targets) as numpy."""
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        trues.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


# ── Plot functions ───────────────────────────────────────────────

def plot_loss_curves(history: dict, save_path: Path):
    """Training & validation loss vs epoch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train")
    ax.plot(epochs, history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path / "loss_curves.png", dpi=150)
    fig.savefig(save_path / "loss_curves.pdf")
    plt.close(fig)
    print(f"  Saved {save_path / 'loss_curves.png'}")


def plot_pred_vs_actual(preds: np.ndarray, trues: np.ndarray, save_path: Path):
    """3x3 scatter grid: predicted vs actual for each species."""
    n_species = preds.shape[1]
    ncols = 3
    nrows = (n_species + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i in range(n_species):
        ax = axes[i]
        ax.scatter(trues[:, i], preds[:, i], s=8, alpha=0.4)
        lo = min(trues[:, i].min(), preds[:, i].min())
        hi = max(trues[:, i].max(), preds[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        r2 = r2_score(trues[:, i], preds[:, i])
        ax.set_title(f"{SPECIES[i]}  (R²={r2:.4f})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n_species, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Predicted vs Actual Concentration", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path / "pred_vs_actual.png", dpi=150, bbox_inches="tight")
    fig.savefig(save_path / "pred_vs_actual.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path / 'pred_vs_actual.png'}")


def plot_r2_bar(preds: np.ndarray, trues: np.ndarray, save_path: Path):
    """Bar chart of R² per species."""
    r2s = [r2_score(trues[:, i], preds[:, i]) for i in range(preds.shape[1])]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(SPECIES[: len(r2s)], r2s, color="steelblue")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² Score")
    ax.set_title("R² per Species (Test Set)")
    for bar, v in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path / "r2_per_species.png", dpi=150)
    fig.savefig(save_path / "r2_per_species.pdf")
    plt.close(fig)
    print(f"  Saved {save_path / 'r2_per_species.png'}")


def plot_error_distribution(preds: np.ndarray, trues: np.ndarray, save_path: Path):
    """Histogram of prediction errors per species."""
    n_species = preds.shape[1]
    ncols = 3
    nrows = (n_species + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i in range(n_species):
        err = preds[:, i] - trues[:, i]
        axes[i].hist(err, bins=40, alpha=0.7, color="steelblue", edgecolor="white")
        axes[i].set_title(f"{SPECIES[i]}  (MAE={np.mean(np.abs(err)):.4f})")
        axes[i].set_xlabel("Prediction Error")
        axes[i].axvline(0, color="red", linestyle="--", linewidth=1)

    for j in range(n_species, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Error Distribution (Test Set)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path / "error_distribution.png", dpi=150, bbox_inches="tight")
    fig.savefig(save_path / "error_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path / 'error_distribution.png'}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained model and generate plots")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt file (default: best_model.pt from config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg["training"]["device"])
    plot_dir = Path(cfg["paths"]["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Loss curves ──────────────────────────────────────────
    history_path = Path(cfg["paths"]["log_dir"]) / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        print("Generating loss curves...")
        plot_loss_curves(history, plot_dir)
    else:
        print(f"Warning: {history_path} not found — skipping loss curves")

    # ── 2. Load model + data ────────────────────────────────────
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

    ckpt = args.checkpoint or str(Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    print(f"Loaded checkpoint: {ckpt}")

    spectra, concentrations = load_spectra_and_concentrations(
        cfg["data"]["spectra_csv"], cfg["data"]["concentration_csv"],
    )
    loaders = build_dataloaders(
        spectra, concentrations,
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        batch_size=cfg["training"]["batch_size"],
        device=str(device),
    )

    # ── 3. Predict on test set ──────────────────────────────────
    preds, trues = predict_all(model, loaders["test"], device)

    # ── 4. Generate all plots ───────────────────────────────────
    print("Generating plots...")
    plot_pred_vs_actual(preds, trues, plot_dir)
    plot_r2_bar(preds, trues, plot_dir)
    plot_error_distribution(preds, trues, plot_dir)

    print(f"\nAll plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
