"""Data loading, preprocessing, and synthetic spectrum generation."""

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, TensorDataset


# ── Load pre-generated CSV datasets ─────────────────────────────

def load_spectra_and_concentrations(spectra_csv: str, concentration_csv: str):
    """Load spectra and concentration matrices from CSV files."""
    spectra = pd.read_csv(spectra_csv).values        # (N, 600)
    concentrations = pd.read_csv(concentration_csv).values  # (N, 7)
    return spectra, concentrations


# ── PNNL cross-section helpers ───────────────────────────────────

def load_pnnl_cross_sections(csv_path: str, gas_names: list[str]):
    """Load PNNL absorbance cross-section data from a two-column-per-gas CSV.

    Returns a dict  {gas_name: (wavenumber_array, absorbance_array)}.
    """
    df = pd.read_csv(csv_path)
    gas_spectra = {}
    for i, name in enumerate(gas_names):
        wn = df.iloc[:, 2 * i].astype(float).values
        ab = df.iloc[:, 2 * i + 1].astype(float).values
        gas_spectra[name] = (wn, ab)
    return gas_spectra


def interpolate_cross_sections(
    gas_spectra: dict,
    gas_names: list[str],
    wn_min: float = 2963.0,
    wn_max: float = 2968.0,
    num_points: int = 600,
    path_length: float = 0.10,
    mole_fractions: dict | None = None,
):
    """Interpolate cross-sections onto a common wavenumber grid.

    Returns:
        interpolated_wn: (num_points,) array
        absorbance_matrix: (num_species, num_points) array
    """
    ln10 = np.log(10)
    interpolated_wn = np.linspace(wn_min, wn_max, num_points)
    rows = []
    for gas in gas_names:
        wn, ab = gas_spectra[gas]
        if mole_fractions is not None:
            ab = ab * mole_fractions[gas] * path_length * ln10
        mask = (wn >= wn_min) & (wn <= wn_max)
        f = interp1d(wn[mask], ab[mask], kind="linear", fill_value="extrapolate")
        rows.append(f(interpolated_wn))
    return interpolated_wn, np.stack(rows)


def generate_synthetic_spectra(
    absorbance_matrix: np.ndarray,
    n_samples: int = 10000,
    num_targets: int = 7,
    seed: int = 42,
):
    """Generate random concentration mixtures and corresponding spectra.

    Args:
        absorbance_matrix: (num_species, num_points)
        n_samples: number of synthetic observations
        num_targets: number of species used as regression targets
        seed: random seed

    Returns:
        spectra: (n_samples, num_points)
        concentrations: (n_samples, num_targets)
    """
    rng = np.random.default_rng(seed)
    num_species = absorbance_matrix.shape[0]
    concentrations = rng.random((n_samples, num_species))
    spectra = concentrations @ absorbance_matrix  # (N, num_points)
    return spectra, concentrations[:, :num_targets]


# ── PyTorch DataLoaders ──────────────────────────────────────────

def build_dataloaders(
    spectra: np.ndarray,
    concentrations: np.ndarray,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    batch_size: int = 16,
    device: str = "cpu",
):
    """Split data and return train / val / test DataLoaders.

    spectra: (N, num_points) – will be reshaped to (N, 1, num_points).
    concentrations: (N, num_targets).
    """
    N = spectra.shape[0]
    spectra = spectra.reshape(N, 1, spectra.shape[1])

    X = torch.tensor(spectra, dtype=torch.float32, device=device)
    Y = torch.tensor(concentrations, dtype=torch.float32, device=device)

    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    splits = {
        "train": (X[:n_train], Y[:n_train]),
        "val": (X[n_train : n_train + n_val], Y[n_train : n_train + n_val]),
        "test": (X[n_train + n_val :], Y[n_train + n_val :]),
    }

    loaders = {}
    for name, (x, y) in splits.items():
        ds = TensorDataset(x, y)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"))
    return loaders
