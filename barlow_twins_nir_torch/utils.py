"""Metrics and visualization utilities (re-exported from barlow_twins_nir.utils)."""

# utils.py has no framework dependency — safe to import directly without
# triggering barlow_twins_nir's __init__.py (which would import TensorFlow).
from barlow_twins_nir.utils import compute_metrics, plot_predictions

__all__ = ['compute_metrics', 'plot_predictions']
