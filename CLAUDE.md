# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research code accompanying the paper **"Barlow Twins for Semi-Supervised Learning in NIR Spectroscopy"**. The project is structured as an installable Python package (`barlow_twins_nir`) with the original notebook kept as a reference and a clean example notebook in `examples/`.

## Package Structure

```
BarlowTwins_NIR/
├── barlow_twins_nir/           # installable package
│   ├── __init__.py             # public API surface
│   ├── data.py                 # loading, filtering, normalization, TF datasets
│   ├── models.py               # encoders, BarlowLoss, BarlowRegressionModel
│   └── utils.py                # metrics and scatter plot visualization
├── examples/
│   └── kiwifruit_example.ipynb # clean demo notebook using the package
├── Barlow_for_NIR_example.ipynb # original reference notebook (read-only)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -e .          # editable install from repo root
# or
pip install -r requirements.txt
```

Core dependencies: `tensorflow>=2.16`, `tensorflow-probability>=0.24`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `livelossplot`.

## Quick Start

```python
from barlow_twins_nir import (
    load_kiwifruit, remove_outliers, normalize_features,
    make_paired_views, make_labeled_dataset, make_semi_supervised_dataset,
    make_validation_dataset, BarlowRegressionModel, build_supervised_model,
    compute_metrics, plot_predictions,
)
```

See `examples/kiwifruit_example.ipynb` for a complete end-to-end walkthrough.

## Original Notebook

`Barlow_for_NIR_example.ipynb` is kept as a read-only reference. It runs on Google Colab (Python 3.12) and requires:
```bash
pip install livelossplot
```

## Architecture

The notebook implements a **semi-supervised regression model** for predicting fruit quality (dry matter content, DM) from NIR spectra with limited labeled samples.

### Data Pipeline
- Input: kiwifruit NIR spectra (`kiwifruit_dat.csv`) with wavelengths X402–X1065 (222 features, 3 nm steps)
- Preprocessing: standard normalization using training set statistics, then Savitzky-Golay 2nd derivative filter (width=13, polyorder=2) applied as a fixed (non-trainable) Conv1D layer
- Outlier removal via Mahalanobis distance on PCA scores
- Data augmentation: repeated paired spectra of the same fruit across multiple NIR devices serve as the two "views" for Barlow Twins

### Model Components

**`build_encoder()`** — The shared backbone:
- Optional fixed Savitzky-Golay Conv1D preprocessing layer (non-trainable)
- Optional trainable Conv1D layer (`conv1`)
- Flatten → Dense layers with configurable sizes

**`BarlowLoss`** — Custom Keras loss class:
- Computes cross-correlation matrix between two encoder outputs using `tfp.stats.correlation`
- Loss = diagonal terms (penalize deviation from 1) + λ × off-diagonal terms (penalize redundancy)
- `lambda_amt` controls the off-diagonal weight (paper uses `1/15`)

**`BarlowRegressionModel`** (main model) — Custom `tf.keras.Model`:
- `train_step` receives zipped `(unlabeled_batch, labeled_batch)` pairs
- Unlabeled data: Barlow loss between two augmented views (different device readings of same fruit)
- Labeled data: MSE regression loss on a small labeled subset
- Consistency loss between predictions of the two views
- Combined weighted loss: `barlow_weight * barlow_loss + mse_weight * regression_loss + consistency_weight * consistency_loss`

**`get_regression_head()`** — Dense layers ending in linear output for regression.

**`build_supervised_model()`** — Builds and compiles a functional Keras model (encoder + regression head) trained with MSE only, for baseline comparison.

### Training Setup
- `make_semi_supervised_dataset()`: zips a large unlabeled `tf.data.Dataset` with a repeated small labeled dataset
- Callbacks: `PlotLossesKerasTF` (live loss plot), `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Optimizer: Adam (lr=0.005) with gradient clipping (`clipvalue=1.0`)
- Model saved to `model4a_weights.keras`

### Baseline Comparison
The notebook also trains a standard supervised model (MSE-only, no Barlow loss) on the same small labeled subset using `build_encoder` + `get_regression_head` as a Keras functional model, for comparison against the semi-supervised approach.

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enc_sizes` | `(16,)` | Encoder hidden layer sizes |
| `reg_sizes` | `[1]` | Regression head layer sizes |
| `nsamp` | `100` | Number of labeled samples (key experimental variable) |
| `barlow_lambda` | `1/15` | Off-diagonal penalty weight in Barlow loss |
| `loss_weight` | `(10.5, 0.5, 0.5, 0.)` | Weights for (Barlow, MSE, consistency, unused) losses |
| `BATCH_SIZE` | `4000` | Batch size for unlabeled data |
| `preprocess` | `3` | Savitzky-Golay derivative order (1=1st deriv, 2=2nd deriv, 3=2nd deriv) |
