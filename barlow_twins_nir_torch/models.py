"""PyTorch model components: Encoder, BarlowLoss, BarlowRegressionModel, SupervisedModel."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import savgol_coeffs


def _activation(name):
    """Return an nn.Module activation by name; 'linear' returns nn.Identity."""
    return {
        'linear': nn.Identity(),
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
    }.get(name, nn.Identity())


class Encoder(nn.Module):
    """Shared CNN + Dense encoder backbone.

    Architecture mirrors the TF ``build_encoder``:
      1. Optional frozen Savitzky-Golay Conv1d preprocessing layer
      2. Optional trainable Conv1d feature-extraction layer
      3. Flatten
      4. Dense (Linear) stack

    Input shape: ``(batch, input_size)`` — 1-D spectra, no channel dim needed.

    Parameters
    ----------
    input_size : int
        Number of wavelength features (default 222).
    layer_sizes : tuple of int
        Sizes of the Linear hidden layers.
    activation : str
        Activation for intermediate Dense layers (default 'relu').
    conv1 : bool
        Add a trainable Conv1d layer (default True).
    preprocess : int
        Savitzky-Golay derivative order. 0 = disabled, 1 = 1st derivative,
        2 or 3 = 2nd derivative (default 3).
    n_filt : int
        Number of filters in the trainable Conv1d (default 50).
    kernel_size : int
        Kernel size for both Conv1d layers (default 13).
    """

    def __init__(self, input_size=222, layer_sizes=(16,), activation='relu',
                 conv1=True, preprocess=3, n_filt=50, kernel_size=13):
        super().__init__()
        self.preprocess = preprocess
        self.use_conv1 = conv1
        self.kernel_size = kernel_size

        # --- Savitzky-Golay frozen Conv1d ---
        if preprocess > 0:
            deriv_order = preprocess - 1  # matches TF: deriv=(preprocess-1)
            sg = savgol_coeffs(kernel_size, 2, deriv=deriv_order, use='conv')
            self.sg_conv = nn.Conv1d(1, 1, kernel_size,
                                     padding=kernel_size // 2, bias=False)
            with torch.no_grad():
                self.sg_conv.weight.copy_(
                    torch.tensor(sg, dtype=torch.float32).view(1, 1, -1)
                )
            self.sg_conv.weight.requires_grad_(False)

        # --- Trainable Conv1d ---
        if conv1:
            in_ch = 1  # input is always (batch, 1, input_size) at this point
            self.conv1_layer = nn.Conv1d(in_ch, n_filt, kernel_size,
                                          padding=kernel_size // 2, bias=False)

        # --- Flat size after conv layers ---
        if conv1:
            flat_size = n_filt * input_size
        else:
            flat_size = input_size  # 1 channel × input_size after flatten

        # --- Dense stack ---
        num_layers = len(layer_sizes)
        dense = []
        current = flat_size
        for i, size in enumerate(layer_sizes):
            dense.append(nn.Linear(current, size))
            # All intermediate layers get activation; last layer is always linear
            if i < num_layers - 1:
                dense.append(_activation(activation))
            current = size
        self.dense = nn.Sequential(*dense)

    def forward(self, x):
        # x: (batch, input_size)
        x = x.unsqueeze(1)                  # (batch, 1, input_size)
        if self.preprocess > 0:
            x = self.sg_conv(x)             # (batch, 1, input_size)
        if self.use_conv1:
            x = self.conv1_layer(x)         # (batch, n_filt, input_size)
        x = x.flatten(1)                    # (batch, n_filt * input_size)
        return self.dense(x)


class RegressionHead(nn.Module):
    """Dense regression head with a final linear output.

    Parameters
    ----------
    input_size : int
        Input feature size (encoder output size).
    reg_sizes : tuple of int
        Layer sizes; the final output is always linear.
    activation : str
        Activation for intermediate layers (default 'linear').
    """

    def __init__(self, input_size, reg_sizes=(1,), activation='linear'):
        super().__init__()
        num = len(reg_sizes)
        layers = []
        current = input_size
        for i, size in enumerate(reg_sizes):
            layers.append(nn.Linear(current, size))
            if i < num - 1:
                layers.append(_activation(activation))
            current = size
        self.layers_ = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers_(x)


class BarlowLoss(nn.Module):
    """Barlow Twins cross-correlation loss (PyTorch).

    Computes the sample cross-correlation matrix between two encoder output
    batches and penalises deviation from an identity matrix.

    Parameters
    ----------
    lambda_amt : float
        Off-diagonal penalty weight (paper uses ``1/15``).
    """

    def __init__(self, lambda_amt=1. / 15):
        super().__init__()
        self.lambda_amt = lambda_amt

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        N = z_a.shape[0]
        # Normalise along batch dimension (matches tfp.stats.correlation)
        z_a = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-8)
        z_b = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-8)

        c = (z_a.T @ z_b) / N  # (D, D) cross-correlation matrix

        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        off_diag = self._off_diagonal(c).pow(2).sum()
        return on_diag + self.lambda_amt * off_diag

    @staticmethod
    def _off_diagonal(c: torch.Tensor) -> torch.Tensor:
        n = c.shape[0]
        return c.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowRegressionModel(nn.Module):
    """Semi-supervised Barlow Twins regression model for NIR spectroscopy.

    Combines Barlow Twins loss on unlabeled paired spectra with MSE regression
    loss on a small labeled subset, plus a consistency loss between the two view
    predictions.

    Use ``compute_loss(unlabeled_batch, labeled_batch)`` inside your training loop,
    or call ``train()`` from ``barlow_twins_nir_torch.train``.

    Parameters
    ----------
    input_size : int
        Encoder input size (number of wavelengths, default 222).
    enc_sizes : tuple of int
        Encoder hidden layer sizes.
    reg_sizes : tuple of int
        Regression head layer sizes.
    loss_weight : tuple of float
        Weights for (barlow, mse, consistency, unused) loss terms.
    barlow_lambda : float
        Off-diagonal penalty in BarlowLoss.
    activation : str
        Encoder Dense activation.
    conv1 : bool
        Enable trainable Conv1d in encoder.
    preprocess : int
        Savitzky-Golay derivative order.
    n_filt : int
        Filters in the trainable Conv1d.
    kernel_size : int
        Kernel size for Conv1d layers.
    """

    def __init__(self, input_size=222, enc_sizes=(16,), reg_sizes=(1,),
                 loss_weight=(10.5, 0.5, 0.5, 0.), barlow_lambda=1. / 15,
                 activation='linear', conv1=True, preprocess=3,
                 n_filt=50, kernel_size=13):
        super().__init__()
        self.encoder = Encoder(input_size, enc_sizes, activation,
                                conv1, preprocess, n_filt, kernel_size)
        self.regression_head = RegressionHead(enc_sizes[-1], reg_sizes)
        self.barlow_loss_fn = BarlowLoss(lambda_amt=barlow_lambda)
        self.loss_weight = loss_weight

    def forward(self, x):
        return self.regression_head(self.encoder(x))

    def compute_loss(self, unlabeled_batch, labeled_batch):
        """Compute combined Barlow + MSE + consistency loss.

        Parameters
        ----------
        unlabeled_batch : tuple of (Tensor, Tensor)
            ``(view_a, view_b)`` — paired unlabeled spectra.
        labeled_batch : tuple of (Tensor, Tensor)
            ``(features, target)`` — small labeled subset.

        Returns
        -------
        loss : Tensor (scalar)
        """
        y_a, y_b = unlabeled_batch
        y_c, target = labeled_batch

        z_a = self.encoder(y_a)
        z_b = self.encoder(y_b)
        z_c = self.encoder(y_c)

        pred_a = self.regression_head(z_a)
        pred_b = self.regression_head(z_b)
        pred_c = self.regression_head(z_c)

        barlow_loss = self.barlow_loss_fn(z_a, z_b)
        mse_loss = F.mse_loss(pred_c, target.unsqueeze(-1))
        consistency_loss = F.mse_loss(pred_a, pred_b)

        return (
            self.loss_weight[0] * barlow_loss
            + self.loss_weight[1] * mse_loss
            + self.loss_weight[2] * consistency_loss
        )


class SupervisedModel(nn.Module):
    """Supervised (MSE-only) baseline model for NIR regression.

    Uses the same encoder + regression head architecture as
    ``BarlowRegressionModel`` but trains with standard MSE on labeled data only.

    Parameters
    ----------
    input_size : int
        Encoder input size (default 222).
    enc_sizes : tuple of int
        Encoder hidden layer sizes.
    reg_sizes : tuple of int
        Regression head layer sizes.
    activation : str
        Encoder Dense activation.
    conv1 : bool
        Enable trainable Conv1d.
    preprocess : int
        Savitzky-Golay derivative order.
    n_filt : int
        Filters in the trainable Conv1d.
    kernel_size : int
        Kernel size for Conv1d layers.
    """

    def __init__(self, input_size=222, enc_sizes=(16,), reg_sizes=(1,),
                 activation='linear', conv1=True, preprocess=3,
                 n_filt=50, kernel_size=13):
        super().__init__()
        self.encoder = Encoder(input_size, enc_sizes, activation,
                                conv1, preprocess, n_filt, kernel_size)
        self.regression_head = RegressionHead(enc_sizes[-1], reg_sizes)

    def forward(self, x):
        return self.regression_head(self.encoder(x))
