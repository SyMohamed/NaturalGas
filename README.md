# Natural Gas Composition Analysis via CNN

Predict natural gas component concentrations (C1–C7) from mid-IR absorbance spectra using a 1-D CNN regressor trained on PNNL cross-section data.

## Project Structure

```
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml            # hyperparameters, data paths
├── src/
│   ├── model.py               # CNNRegressor architecture
│   ├── train.py               # training / evaluation loop
│   ├── data.py                # data loading, preprocessing, synthetic generation
│   └── utils.py               # seeding, device helpers
├── scripts/
│   └── submit_ibex.sh         # SLURM job script for IBEX
├── data/                      # CSV data files (gitignored – add manually)
└── experiments/               # checkpoints, logs, plots (gitignored)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Place your data CSVs in data/
#   - standard_simulated_spectra_interp_excess.csv
#   - standard_simulated_concentrations_interp_excess.csv
#   - PNNL_CS.csv

# Train
python -m src.train --config configs/config.yaml
```

## IBEX (SLURM)

```bash
sbatch scripts/submit_ibex.sh
```
