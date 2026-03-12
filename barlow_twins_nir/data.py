"""Data loading, preprocessing, and TensorFlow dataset utilities for NIR spectroscopy."""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler


def load_kiwifruit(filepath, dm_cutoff=7, min_readings=2):
    """Load and filter kiwifruit NIR dataset.

    Parameters
    ----------
    filepath : str
        Path to the kiwifruit_dat.csv file.
    dm_cutoff : float
        Minimum DM value to keep (default 7).
    min_readings : int
        Minimum number of readings per sample_id to keep (default 2).

    Returns
    -------
    pd.DataFrame
        Filtered and date-sorted kiwifruit DataFrame.
    """
    kiwi = pd.read_csv(filepath)
    kiwi = kiwi.dropna(subset=['SSC'])
    kiwi = kiwi[kiwi['DM'] > dm_cutoff]
    kiwi['Date'] = pd.to_datetime(kiwi['Date'], format='%d/%m/%Y')
    kiwi = kiwi.sort_values(by=['Date'], kind='stable')
    kiwi = kiwi.groupby('sample_id').filter(lambda x: len(x) >= min_readings)
    return kiwi


def remove_outliers(df, x_lower='X402', x_upper='X1065', n_components=20, threshold=1200):
    """Remove spectral outliers using PCA + Mahalanobis distance.

    Applies standard normalization → Savitzky-Golay 2nd derivative → PCA →
    MinCovDet Mahalanobis distance, then discards rows above threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Kiwifruit DataFrame with a 'Dataset' column.
    x_lower : str
        First wavelength column (default 'X402').
    x_upper : str
        Last wavelength column (default 'X1065').
    n_components : int
        Number of PCA components to use for Mahalanobis (default 20).
    threshold : float
        Mahalanobis distance threshold for outlier removal (default 1200).

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers removed.
    """
    features = df.loc[:, x_lower:x_upper]
    train_mask = df['Dataset'] == 'Training'
    features_norm = (features - features.loc[train_mask].mean()) / features.loc[train_mask].std()

    X_sg = savgol_filter(features_norm, 13, polyorder=2, deriv=2)

    pca = PCA()
    T = pca.fit_transform(StandardScaler().fit_transform(X_sg))

    robust_cov = MinCovDet().fit(T[:, :n_components])
    m = robust_cov.mahalanobis(T[:, :n_components])

    return df[m < threshold]


def normalize_features(features, train_mask):
    """Normalize features using training set mean and std.

    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame (wavelength columns).
    train_mask : pd.Series or array-like of bool
        Boolean mask selecting training rows.

    Returns
    -------
    pd.DataFrame
        Normalized features: (X - train_mean) / train_std.
    """
    train_mean = features.loc[train_mask].mean()
    train_std = features.loc[train_mask].std()
    return (features - train_mean) / train_std


def make_paired_views(df, x_lower='X402', x_upper='X1065'):
    """Build paired NIR spectra from multiple device readings of the same fruit.

    Each sample is measured by multiple NIR devices; this function constructs
    two aligned DataFrames (view A and view B) where each row pair corresponds
    to two different device readings of the same fruit.

    Parameters
    ----------
    df : pd.DataFrame
        Kiwifruit DataFrame (after outlier removal).
    x_lower : str
        First wavelength column name (default 'X402').
    x_upper : str
        Last wavelength column name (default 'X1065').

    Returns
    -------
    repeated_kiwi_x : pd.DataFrame
        View A — each original row repeated by its group size, sorted by sample_id.
    repeated_kiwi_y : pd.DataFrame
        View B — each group block repeated by its group size, sorted by sample_id.
    """
    kiwi_sorted = df.sort_values(by=['sample_id'])

    # View A: each row repeated by the size of its sample_id group
    repeated_kiwi_x = kiwi_sorted.loc[
        kiwi_sorted.index.repeat(kiwi_sorted.groupby('sample_id').transform('size'))
    ]

    # View B: each group block repeated 'count' times in full
    group_counts = df.groupby('sample_id').size().to_dict()
    repeated_rows = []
    for group_name, group_df in df.groupby('sample_id'):
        count = group_counts[group_name]
        for _ in range(count):
            repeated_rows.append(group_df)
    repeated_kiwi_y = pd.concat(repeated_rows, ignore_index=True)

    # Align and reset indices
    repeated_kiwi_x = repeated_kiwi_x.sort_values(by=['sample_id'], kind='stable')
    repeated_kiwi_x = repeated_kiwi_x.reset_index(drop=True)
    repeated_kiwi_y = repeated_kiwi_y.reset_index(drop=True)

    return repeated_kiwi_x, repeated_kiwi_y


def make_labeled_dataset(kiwi, nsamp, x_lower='X402', x_upper='X1065',
                         target_col='DM', batch_size=3000):
    """Create a labeled TF dataset from a small subset of training samples.

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
        Batch size for the returned dataset (default 3000).

    Returns
    -------
    dataset_label : tf.data.Dataset
        Batched dataset of (normalized_features, targets).
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

    dataset_label = tf.data.Dataset.from_tensor_slices(
        (features_sub_norm, kiwi_sub[target_col])
    )
    dataset_label = dataset_label.batch(min(len(kiwi_sub), batch_size))

    return dataset_label, tail_sample_ids, features_sub_norm, kiwi_sub[target_col]


def make_semi_supervised_dataset(unlabeled_ds, labeled_ds, n_repeats=None,
                                 labeled_batch_size=32, unlabeled_batch_size=128):
    """Zip unlabeled and labeled datasets for semi-supervised training.

    The labeled dataset is shuffled and repeated so it pairs with every
    unlabeled batch; the unlabeled dataset is shuffled once per epoch.

    Parameters
    ----------
    unlabeled_ds : tf.data.Dataset
        Batched unlabeled dataset yielding (view_a, view_b) pairs.
    labeled_ds : tf.data.Dataset
        Batched labeled dataset yielding (features, targets).
    n_repeats : int, optional
        Unused; kept for API compatibility.

    Returns
    -------
    tf.data.Dataset
        Zipped dataset yielding ((view_a, view_b), (features, targets)).
    """
    labeled_ds = labeled_ds.shuffle(1000).repeat()
    unlabeled_ds = unlabeled_ds.shuffle(1000)
    return tf.data.Dataset.zip((unlabeled_ds, labeled_ds))


def make_validation_dataset(features_x_norm, features_y_norm,
                             repeated_kiwi_x, repeated_kiwi_y,
                             target_col='DM', batch_size=4000):
    """Build validation and test TF datasets from paired-view normalized features.

    Parameters
    ----------
    features_x_norm : pd.DataFrame
        Normalized features for view A (from make_paired_views + normalize_features).
    features_y_norm : pd.DataFrame
        Normalized features for view B.
    repeated_kiwi_x : pd.DataFrame
        Metadata DataFrame for view A (must have 'Dataset' and target_col columns).
    repeated_kiwi_y : pd.DataFrame
        Metadata DataFrame for view B.
    target_col : str
        Target column name (default 'DM').
    batch_size : int
        Batch size (default 4000).

    Returns
    -------
    dataset_val : tf.data.Dataset
        Validation dataset yielding (x_features, y_features, targets).
    dataset_test : tf.data.Dataset
        Non-training dataset (validation + test) yielding (x_features, y_features, targets).
    """
    val_mask_x = repeated_kiwi_x['Dataset'] == 'Validation'
    val_mask_y = repeated_kiwi_y['Dataset'] == 'Validation'
    dataset_val = tf.data.Dataset.from_tensor_slices((
        features_x_norm.loc[val_mask_x].values,
        features_y_norm.loc[val_mask_y].values,
        repeated_kiwi_x.loc[val_mask_x, target_col],
    ))
    dataset_val = dataset_val.shuffle(buffer_size=int(val_mask_x.sum()))
    dataset_val = dataset_val.batch(batch_size)

    nontr_mask_x = repeated_kiwi_x['Dataset'] != 'Training'
    nontr_mask_y = repeated_kiwi_y['Dataset'] != 'Training'
    dataset_test = tf.data.Dataset.from_tensor_slices((
        features_x_norm.loc[nontr_mask_x].values,
        features_y_norm.loc[nontr_mask_y].values,
        repeated_kiwi_x.loc[nontr_mask_x, target_col],
    ))
    dataset_test = dataset_test.shuffle(buffer_size=int(nontr_mask_x.sum()))
    dataset_test = dataset_test.batch(batch_size)

    return dataset_val, dataset_test
