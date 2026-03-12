"""Barlow Twins semi-supervised learning for NIR spectroscopy."""

from .data import (
    load_kiwifruit,
    remove_outliers,
    normalize_features,
    make_paired_views,
    make_labeled_dataset,
    make_semi_supervised_dataset,
    make_validation_dataset,
)
from .models import (
    build_encoder,
    get_regression_head,
    BarlowLoss,
    BarlowRegressionModel,
    build_supervised_model,
)
from .utils import compute_metrics, plot_predictions

__all__ = [
    # data
    'load_kiwifruit',
    'remove_outliers',
    'normalize_features',
    'make_paired_views',
    'make_labeled_dataset',
    'make_semi_supervised_dataset',
    'make_validation_dataset',
    # models
    'build_encoder',
    'get_regression_head',
    'BarlowLoss',
    'BarlowRegressionModel',
    'build_supervised_model',
    # utils
    'compute_metrics',
    'plot_predictions',
]
