# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research code accompanying the paper **"Barlow Twins for Semi-Supervised Learning in NIR Spectroscopy"**. The project is structured as an installable Python package (`barlow_twins_nir`) with the original notebook kept as a reference and a clean example notebook in `examples/`.

## Package Structure

```
BarlowTwins_NIR/
├── barlow_twins_nir/               # TensorFlow backend
│   ├── __init__.py
│   ├── data.py                     # loading, filtering, normalization, tf.data datasets
│   │                               # NOTE: TF imports are lazy — pure helpers are
│   │                               # safely re-imported by barlow_twins_nir_torch
│   ├── models.py                   # build_encoder, BarlowLoss, BarlowRegressionModel
│   └── utils.py                    # compute_metrics, plot_predictions (no TF dep)
├── barlow_twins_nir_torch/         # PyTorch backend
│   ├── __init__.py
│   ├── data.py                     # Dataset classes, DataLoader factories
│   │                               # re-exports pure helpers from barlow_twins_nir.data
│   ├── models.py                   # Encoder, BarlowLoss, BarlowRegressionModel, SupervisedModel
│   ├── train.py                    # train_barlow(), train_supervised()
│   └── utils.py                    # re-exports from barlow_twins_nir.utils
├── examples/
│   ├── kiwifruit_example.ipynb          # TF end-to-end demo
│   └── kiwifruit_example_torch.ipynb   # PyTorch end-to-end demo
├── Barlow_for_NIR_example.ipynb    # original reference notebook (read-only)
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Shared data utilities

`load_kiwifruit`, `remove_outliers`, `normalize_features`, and `make_paired_views`
live in `barlow_twins_nir/data.py` and have **no TensorFlow dependency** (TF is
imported lazily only inside the `tf.data`-returning functions). Both packages
import these functions directly from that module.

## Installation

```bash
# TensorFlow backend
pip install -e .[tensorflow]

# PyTorch backend
pip install -e .[torch]

# Both
pip install -e .[all]
```

## Quick Start

```python
# TensorFlow
from barlow_twins_nir import (
    load_kiwifruit, remove_outliers, normalize_features, make_paired_views,
    make_labeled_dataset, make_semi_supervised_dataset, make_validation_dataset,
    BarlowRegressionModel, build_supervised_model, compute_metrics, plot_predictions,
)

# PyTorch
from barlow_twins_nir_torch import (
    load_kiwifruit, remove_outliers, normalize_features, make_paired_views,
    PairedNIRDataset, LabeledNIRDataset, make_labeled_dataset,
    make_validation_dataloaders, BarlowRegressionModel, SupervisedModel,
    train_barlow, train_supervised, compute_metrics, plot_predictions,
)
```

See `examples/kiwifruit_example.ipynb` (TF) and `examples/kiwifruit_example_torch.ipynb` (PyTorch).

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

### PyTorch equivalents (`barlow_twins_nir_torch`)

- `Encoder` — `nn.Module` matching the TF encoder (frozen SG Conv1d + trainable Conv1d + Linear stack)
- `BarlowLoss` — `nn.Module`; uses normalized dot-product correlation (equivalent to `tfp.stats.correlation`)
- `BarlowRegressionModel` — `nn.Module` with a `compute_loss(unlabeled, labeled)` method
- `SupervisedModel` — `nn.Module` (encoder + regression head, MSE training)
- `train_barlow()` / `train_supervised()` — training loops with Adam, `ReduceLROnPlateau`, early stopping, and best-weights checkpointing; `train_barlow` uses `itertools.cycle` to pair labeled/unlabeled batches

### Training Setup
- **TF:** `make_semi_supervised_dataset()` zips a large unlabeled `tf.data.Dataset` with a repeated small labeled dataset; callbacks: `PlotLossesKerasTF`, `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- **PyTorch:** `train_barlow()` iterates `zip(unlabeled_loader, cycle(labeled_loader))`; both functions handle LR scheduling, early stopping, and checkpointing internally
- Optimizer: Adam (lr=0.005), gradient clipping (clipvalue=1.0)

### Baseline Comparison
Both backends train a supervised (MSE-only) model on the same small labeled subset using the same encoder + regression head architecture, for direct comparison against the semi-supervised Barlow Twins approach.

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
