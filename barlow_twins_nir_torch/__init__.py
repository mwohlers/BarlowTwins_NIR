"""Barlow Twins semi-supervised learning for NIR spectroscopy — PyTorch backend."""

from .data import (
    load_kiwifruit,
    remove_outliers,
    normalize_features,
    make_paired_views,
    PairedNIRDataset,
    LabeledNIRDataset,
    PairedValidationDataset,
    make_labeled_dataset,
    make_validation_dataloaders,
)
from .models import (
    Encoder,
    RegressionHead,
    BarlowLoss,
    BarlowRegressionModel,
    SupervisedModel,
)
from .train import train_barlow, train_supervised
from .utils import compute_metrics, plot_predictions

__all__ = [
    # data
    'load_kiwifruit',
    'remove_outliers',
    'normalize_features',
    'make_paired_views',
    'PairedNIRDataset',
    'LabeledNIRDataset',
    'PairedValidationDataset',
    'make_labeled_dataset',
    'make_validation_dataloaders',
    # models
    'Encoder',
    'RegressionHead',
    'BarlowLoss',
    'BarlowRegressionModel',
    'SupervisedModel',
    # training
    'train_barlow',
    'train_supervised',
    # utils
    'compute_metrics',
    'plot_predictions',
]
