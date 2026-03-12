"""Training utilities for BarlowRegressionModel and SupervisedModel."""

import math
from itertools import cycle

import torch
import torch.nn.functional as F


def train_barlow(model, unlabeled_loader, labeled_loader, val_loader=None,
                 epochs=1000, lr=0.005, clip_value=1.0,
                 patience=50, min_delta=1e-3,
                 lr_patience=25, lr_factor=0.5, min_lr=1e-6,
                 checkpoint_path='model_barlow.pt', device=None, verbose=True):
    """Train a BarlowRegressionModel with early stopping and LR scheduling.

    Mirrors the TF training setup: Adam optimizer, ReduceLROnPlateau,
    EarlyStopping (on training loss), and best-weights checkpointing.

    The labeled DataLoader is cycled so it pairs with every unlabeled batch,
    matching the ``tf.data.Dataset.zip`` + ``labeled_ds.repeat()`` behaviour.

    Parameters
    ----------
    model : BarlowRegressionModel
    unlabeled_loader : DataLoader
        Yields ``(view_a, view_b)`` batches.
    labeled_loader : DataLoader
        Yields ``(features, target)`` batches.
    val_loader : DataLoader, optional
        Yields ``(view_a, view_b, target)`` batches; used only for logging.
    epochs : int
    lr : float
    clip_value : float or None
        Gradient clipping value (default 1.0).
    patience : int
        Early-stopping patience (epochs without improvement in train loss).
    min_delta : float
        Minimum improvement to reset early-stopping counter.
    lr_patience : int
        ReduceLROnPlateau patience.
    lr_factor : float
        ReduceLROnPlateau decay factor.
    min_lr : float
        Minimum learning rate.
    checkpoint_path : str
        Path to save best model weights.
    device : str or torch.device or None
        Defaults to CUDA if available, otherwise CPU.
    verbose : bool
        Print progress every 10 epochs.

    Returns
    -------
    history : dict
        ``{'loss': [...], 'val_loss': [...]}``
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=lr_patience, factor=lr_factor, min_lr=min_lr,
    )

    best_loss = math.inf
    no_improve = 0
    history = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.
        n_batches = 0

        for unlabeled_batch, labeled_batch in zip(unlabeled_loader,
                                                   cycle(labeled_loader)):
            unlabeled_batch = [t.to(device) for t in unlabeled_batch]
            labeled_batch = [t.to(device) for t in labeled_batch]

            optimizer.zero_grad()
            loss = model.compute_loss(unlabeled_batch, labeled_batch)
            loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        history['loss'].append(epoch_loss)
        scheduler.step(epoch_loss)

        val_loss = None
        if val_loader is not None:
            val_loss = _evaluate_barlow(model, val_loader, device)
            history['val_loss'].append(val_loss)

        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1

        if verbose and (epoch + 1) % 10 == 0:
            msg = f'Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}'
            if val_loss is not None:
                msg += f'  val_loss={val_loss:.4f}'
            print(msg)

        if no_improve >= patience:
            if verbose:
                print(f'Early stopping at epoch {epoch + 1}')
            break

    # Restore best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history


def train_supervised(model, train_loader, val_data=None,
                     epochs=1000, lr=0.005, clip_value=None,
                     patience=150, min_delta=1e-3,
                     lr_patience=25, lr_factor=0.5, min_lr=1e-6,
                     checkpoint_path='model_supervised.pt', device=None,
                     verbose=True):
    """Train a SupervisedModel with early stopping and LR scheduling.

    Parameters
    ----------
    model : SupervisedModel
    train_loader : DataLoader
        Yields ``(features, target)`` batches.
    val_data : tuple of (Tensor, Tensor) or None
        ``(val_features, val_targets)`` for validation loss logging.
    epochs : int
    lr : float
    clip_value : float or None
        Gradient clipping value (default None = disabled).
    patience : int
        Early-stopping patience on training loss.
    min_delta : float
    lr_patience : int
    lr_factor : float
    min_lr : float
    checkpoint_path : str
    device : str or torch.device or None
    verbose : bool

    Returns
    -------
    history : dict
        ``{'loss': [...], 'val_loss': [...]}``
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=lr_patience, factor=lr_factor, min_lr=min_lr,
    )

    if val_data is not None:
        val_x = torch.tensor(val_data[0], dtype=torch.float32).to(device) \
            if not isinstance(val_data[0], torch.Tensor) else val_data[0].to(device)
        val_y = torch.tensor(val_data[1], dtype=torch.float32).to(device) \
            if not isinstance(val_data[1], torch.Tensor) else val_data[1].to(device)

    best_loss = math.inf
    no_improve = 0
    history = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.
        n_batches = 0

        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(features).squeeze(-1)
            loss = F.mse_loss(pred, target)
            loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        history['loss'].append(epoch_loss)
        scheduler.step(epoch_loss)

        val_loss = None
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x).squeeze(-1)
                val_loss = F.mse_loss(val_pred, val_y).item()
            history['val_loss'].append(val_loss)

        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1

        if verbose and (epoch + 1) % 10 == 0:
            msg = f'Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}'
            if val_loss is not None:
                msg += f'  val_loss={val_loss:.4f}'
            print(msg)

        if no_improve >= patience:
            if verbose:
                print(f'Early stopping at epoch {epoch + 1}')
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history


def _evaluate_barlow(model, val_loader, device):
    """Compute mean MSE on a PairedValidationDataset loader."""
    model.eval()
    total = 0.
    n = 0
    with torch.no_grad():
        for x_a, _x_b, target in val_loader:
            x_a, target = x_a.to(device), target.to(device)
            pred = model(x_a).squeeze(-1)
            total += F.mse_loss(pred, target).item()
            n += 1
    return total / max(n, 1)
