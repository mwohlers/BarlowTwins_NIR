"""Metrics and visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    """Compute regression metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    dict
        Dictionary with keys 'rmse', 'r2', 'mse'.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'rmse': np.sqrt(mse),
        'r2': r2_score(y_true, y_pred),
        'mse': mse,
    }


def plot_predictions(y_true, y_pred,
                     title='Model Predictions vs Observed Values',
                     xlabel='Predicted DMC (%)',
                     ylabel='Observed DMC (%)',
                     save_path=None):
    """Scatter plot of predicted vs observed values with metrics overlay.

    Draws a perfect-prediction line (y=x) and a linear fit line, and
    annotates with RMSE and R².

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Model predictions.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    save_path : str or None
        If provided, save the figure to this path at 600 dpi.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='none')

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect prediction')

    z = np.polyfit(y_pred, y_true, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), 'b-', alpha=0.5, linewidth=2, label='Linear fit')

    textstr = f"RMSE = {metrics['rmse']:.2f}\nR² = {metrics['r2']:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')

    return fig
