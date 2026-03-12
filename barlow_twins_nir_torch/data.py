"""PyTorch Dataset classes and DataLoader factories for NIR spectroscopy.

Pure data-wrangling helpers (load_kiwifruit, remove_outliers, normalize_features,
make_paired_views) are re-exported from barlow_twins_nir.data — they have no
TensorFlow dependency.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Re-export framework-agnostic helpers so users only need one import
from barlow_twins_nir.data import (
    load_kiwifruit,
    remove_outliers,
    normalize_features,
    make_paired_views,
)


class PairedNIRDataset(Dataset):
    """Unlabeled dataset of paired NIR spectra (view_a, view_b).

    Each sample is a pair of normalized spectra from two different NIR device
    readings of the same fruit, used as the two augmented views for Barlow Twins.

    Parameters
    ----------
    features_x_norm : pd.DataFrame
        Normalized features for view A.
    features_y_norm : pd.DataFrame
        Normalized features for view B (same index alignment as x).
    mask : array-like of bool, optional
        Row mask to select a subset (e.g. training split).
    """

    def __init__(self, features_x_norm, features_y_norm, mask=None):
        if mask is not None:
            features_x_norm = features_x_norm.loc[mask]
            features_y_norm = features_y_norm.loc[mask]
        self.x = torch.tensor(features_x_norm.values, dtype=torch.float32)
        self.y = torch.tensor(features_y_norm.values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LabeledNIRDataset(Dataset):
    """Labeled dataset of NIR spectra with regression targets.

    Parameters
    ----------
    features_norm : pd.DataFrame or np.ndarray
        Normalized feature matrix.
    targets : pd.Series or np.ndarray
        Regression target values.
    """

    def __init__(self, features_norm, targets):
        if hasattr(features_norm, 'values'):
            features_norm = features_norm.values
        if hasattr(targets, 'values'):
            targets = targets.values
        self.x = torch.tensor(features_norm, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PairedValidationDataset(Dataset):
    """Validation/test dataset with paired views and regression targets.

    Mirrors the TF validation dataset structure: each sample is
    (view_a, view_b, target), consumed by BarlowRegressionModel.evaluate().

    Parameters
    ----------
    features_x_norm : pd.DataFrame
        View A normalized features.
    features_y_norm : pd.DataFrame
        View B normalized features.
    targets : pd.Series or np.ndarray
        Regression targets.
    mask : array-like of bool, optional
        Row mask to select a subset.
    """

    def __init__(self, features_x_norm, features_y_norm, targets, mask=None):
        if mask is not None:
            features_x_norm = features_x_norm.loc[mask]
            features_y_norm = features_y_norm.loc[mask]
            if hasattr(targets, 'loc'):
                targets = targets.loc[mask]
            else:
                targets = targets[mask]
        self.x = torch.tensor(features_x_norm.values, dtype=torch.float32)
        self.y = torch.tensor(features_y_norm.values, dtype=torch.float32)
        if hasattr(targets, 'values'):
            targets = targets.values
        self.t = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.t[idx]


def make_labeled_dataset(kiwi, nsamp, x_lower='X402', x_upper='X1065',
                         target_col='DM', batch_size=3000):
    """Create a labeled DataLoader from a small subset of training samples.

    Selects the last ``nsamp`` unique sample_ids from the training split and
    normalizes using all training data statistics.

    Parameters
    ----------
    kiwi : pd.DataFrame
        Full kiwifruit DataFrame (with 'Dataset' column).
    nsamp : int
        Number of unique sample_ids to use as labeled data.
    x_lower : str
        First wavelength column (default 'X402').
    x_upper : str
        Last wavelength column (default 'X1065').
    target_col : str
        Target column name (default 'DM').
    batch_size : int
        Batch size for the returned DataLoader (default 3000).

    Returns
    -------
    loader : DataLoader
        Labeled DataLoader yielding (features, targets) batches.
    tail_sample_ids : np.ndarray
        The selected sample IDs.
    features_sub_norm : pd.DataFrame
        Normalized features for the labeled subset.
    labels : pd.Series
        Target values for the labeled subset.
    """
    kiwi_train = kiwi[kiwi['Dataset'] == 'Training']
    unique_sample_ids = kiwi_train['sample_id'].unique()
    tail_sample_ids = unique_sample_ids[-nsamp:]

    kiwi_sub = kiwi_train[kiwi_train['sample_id'].isin(tail_sample_ids)]
    features_train = kiwi_train.loc[:, x_lower:x_upper]
    features_sub = kiwi_sub.loc[:, x_lower:x_upper]
    features_sub_norm = (features_sub - features_train.mean()) / features_train.std()

    dataset = LabeledNIRDataset(features_sub_norm, kiwi_sub[target_col])
    loader = DataLoader(dataset, batch_size=min(len(dataset), batch_size), shuffle=True)

    return loader, tail_sample_ids, features_sub_norm, kiwi_sub[target_col]


def make_validation_dataloaders(features_x_norm, features_y_norm,
                                repeated_kiwi_x, repeated_kiwi_y,
                                target_col='DM', batch_size=4000):
    """Build validation and test DataLoaders from paired-view normalized features.

    Parameters
    ----------
    features_x_norm : pd.DataFrame
        Normalized features for view A.
    features_y_norm : pd.DataFrame
        Normalized features for view B.
    repeated_kiwi_x : pd.DataFrame
        Metadata for view A (must have 'Dataset' and target_col columns).
    repeated_kiwi_y : pd.DataFrame
        Metadata for view B.
    target_col : str
        Target column name (default 'DM').
    batch_size : int
        Batch size (default 4000).

    Returns
    -------
    val_loader : DataLoader
        Validation DataLoader yielding (view_a, view_b, target).
    test_loader : DataLoader
        Non-training DataLoader yielding (view_a, view_b, target).
    """
    val_mask = repeated_kiwi_x['Dataset'] == 'Validation'
    val_ds = PairedValidationDataset(
        features_x_norm, features_y_norm,
        repeated_kiwi_x.loc[val_mask, target_col], mask=val_mask,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    nontr_mask = repeated_kiwi_x['Dataset'] != 'Training'
    test_ds = PairedValidationDataset(
        features_x_norm, features_y_norm,
        repeated_kiwi_x.loc[nontr_mask, target_col], mask=nontr_mask,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return val_loader, test_loader
