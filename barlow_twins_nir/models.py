"""Model components: encoder, regression head, Barlow loss, and full model classes."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Reshape
from scipy.signal import savgol_coeffs


def build_encoder(init_shape=(222,), layer_sizes=(16,), activation='relu',
                  conv1=1., preprocess=3, n_filt=50, kernel_size=13, l2=0.0,
                  initializer=None, last_activation='linear'):
    """Build the shared CNN+Dense encoder backbone.

    Optionally prepends a frozen Savitzky-Golay Conv1D derivative filter
    and/or a trainable Conv1D feature-extraction layer before the Dense stack.

    Parameters
    ----------
    init_shape : tuple of int
        Input shape, e.g. ``(222,)`` for 222 wavelength features.
    layer_sizes : tuple of int
        Sizes of the Dense hidden layers.
    activation : str
        Activation function for Dense layers (default 'relu').
    conv1 : float
        If > 0, add a trainable Conv1D layer (default 1.0 = enabled).
    preprocess : int
        Savitzky-Golay derivative order. 0 = disabled, 1 = 1st derivative,
        2 or 3 = 2nd derivative (default 3).
    n_filt : int
        Number of filters in the trainable Conv1D layer (default 50).
    kernel_size : int
        Kernel size for both Conv1D layers (default 13).
    l2 : float
        L2 regularization weight for Dense layers (default 0.0).
    initializer : keras initializer or None
        Weight initializer; defaults to ``HeNormal(seed=123)``.
    last_activation : str
        Activation for the final Dense layer (default 'linear').

    Returns
    -------
    encoder : keras.Sequential
        Uncompiled encoder model.
    """
    if initializer is None:
        initializer = tf.keras.initializers.HeNormal(seed=123)

    num_layers = len(layer_sizes)
    encoder = keras.Sequential(name='encoder')
    encoder.add(keras.Input(init_shape))
    encoder.add(Reshape((init_shape[0], 1), input_shape=(init_shape,)))

    if preprocess > 0.0:
        sg = savgol_coeffs(kernel_size, 2, deriv=(preprocess - 1), use='conv')
        encoder.add(Conv1D(filters=1, strides=1, kernel_size=kernel_size,
                           padding='same', activation='linear',
                           kernel_initializer=initializer, use_bias=False))
        encoder.layers[1].set_weights(
            np.array([np.expand_dims(np.transpose(np.array([sg])), axis=1)])
        )
        encoder.layers[1].trainable = False

    if conv1 > 0.0:
        encoder.add(Conv1D(filters=n_filt, strides=1, kernel_size=kernel_size,
                           padding='same', activation='linear',
                           kernel_initializer=initializer, use_bias=False))

    encoder.add(Flatten())

    for i in range(num_layers):
        act = activation if i < num_layers - 1 else last_activation
        encoder.add(Dense(units=layer_sizes[i], activation=act,
                          kernel_regularizer=tf.keras.regularizers.l2(l2)))

    return encoder


def get_regression_head(encoder_output_size, reg_sizes=(1,), activation='linear',
                        initializer=None):
    """Build the regression head Dense stack.

    Parameters
    ----------
    encoder_output_size : int or tuple
        Output size of the encoder (used as input shape).
    reg_sizes : tuple of int
        Sizes of the Dense layers; the final layer always has linear activation.
    activation : str
        Activation for intermediate layers (default 'linear').
    initializer : keras initializer or None
        Defaults to ``HeNormal(seed=123)``.

    Returns
    -------
    regression_head : keras.Sequential
    """
    if initializer is None:
        initializer = tf.keras.initializers.HeNormal(seed=123)

    if isinstance(encoder_output_size, (tuple, list)):
        input_shape = (encoder_output_size[-1],)
    else:
        input_shape = (encoder_output_size,)

    num_reg_layers = len(reg_sizes)
    regression_head = keras.Sequential(name='regression_head')
    regression_head.add(keras.Input(input_shape))

    for i, size in enumerate(reg_sizes):
        act = activation if i < num_reg_layers - 1 else 'linear'
        regression_head.add(Dense(size, activation=act,
                                  kernel_initializer=initializer))

    return regression_head


class BarlowLoss(tf.keras.losses.Loss):
    """Barlow Twins cross-correlation loss.

    Penalizes deviation of the cross-correlation matrix from an identity
    matrix: diagonal terms should be 1 (invariance), off-diagonal terms
    should be 0 (redundancy reduction).

    Parameters
    ----------
    batch_size : int
        Batch size; used for numerical scaling.
    lambda_amt : float
        Weight for the off-diagonal penalty (paper uses ``1/15``).
    """

    def __init__(self, batch_size: int, lambda_amt: float = 1.0):
        super().__init__()
        self.lambda_amt = lambda_amt
        self.batch_size = batch_size

    def get_off_diag(self, c: tf.Tensor) -> tf.Tensor:
        zero_diag = tf.zeros(c.shape[-1])
        return tf.linalg.set_diag(c, zero_diag)

    def cross_corr_matrix_loss(self, c: tf.Tensor) -> tf.Tensor:
        c_diff = tf.pow(tf.linalg.diag_part(c) - 1, 2)
        off_diag = tf.pow(self.get_off_diag(c), 2) * self.lambda_amt
        return tf.reduce_sum(c_diff) + tf.reduce_sum(off_diag)

    def call(self, z_a: tf.Tensor, z_b: tf.Tensor) -> tf.Tensor:
        c = tfp.stats.correlation(z_a, z_b, sample_axis=0, event_axis=-1)
        return self.cross_corr_matrix_loss(c)


class BarlowRegressionModel(tf.keras.Model):
    """Semi-supervised Barlow Twins regression model for NIR spectroscopy.

    Combines Barlow Twins contrastive loss on unlabeled paired spectra with
    MSE regression loss on a small labeled subset.

    Parameters
    ----------
    init_shape : tuple of int
        Encoder input shape (default ``(222,)``).
    enc_sizes : tuple of int
        Encoder hidden layer sizes (default ``(16,)``).
    reg_sizes : list of int
        Regression head layer sizes (default ``[1]``).
    loss_weight : tuple of float
        Weights for (barlow, mse, consistency, unused) losses.
    barlow_lambda : float
        Off-diagonal penalty in BarlowLoss (default ``1/15``).
    BATCH_SIZE : int
        Unlabeled batch size passed to BarlowLoss (default 4000).
    activation : str
        Encoder Dense activation (default 'linear').
    activation_reg : str
        Regression head intermediate activation (default 'linear').
    conv1 : float
        Enable trainable Conv1D in encoder if > 0 (default 1.0).
    preprocess : int
        Savitzky-Golay derivative order for encoder (default 3).
    initializer : keras initializer or None
        Defaults to ``LecunNormal(seed=123)``.
    """

    def __init__(self, init_shape=(222,), enc_sizes=(16,), reg_sizes=(1,),
                 loss_weight=(10.5, 0.5, 0.5, 0.), barlow_lambda=1. / 15,
                 BATCH_SIZE=4000, activation='linear', activation_reg='linear',
                 conv1=1., preprocess=3, initializer=None):
        super().__init__()
        if initializer is None:
            initializer = tf.keras.initializers.LecunNormal(seed=123)

        self.encoder_a = build_encoder(
            init_shape=init_shape, layer_sizes=enc_sizes,
            activation=activation, conv1=conv1, preprocess=preprocess,
            initializer=initializer,
        )
        self.regression_head = get_regression_head(
            encoder_output_size=enc_sizes, reg_sizes=reg_sizes,
            activation=activation_reg, initializer=initializer,
        )
        self.barlow_loss = BarlowLoss(batch_size=BATCH_SIZE, lambda_amt=barlow_lambda)
        self.loss_weight = loss_weight
        self.loss_tracker = keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, batch):
        batch1, batch2 = batch
        y_a, y_b = batch1
        y_c, regression_target2 = batch2

        with tf.GradientTape() as tape:
            z_a = self.encoder_a(y_a, training=True)
            regression_output_a = self.regression_head(z_a, training=True)
            z_b = self.encoder_a(y_b, training=True)
            regression_output_b = self.regression_head(z_b, training=True)
            z_c = self.encoder_a(y_c, training=True)
            regression_output_c = self.regression_head(z_c, training=True)

            consistency_loss = tf.reduce_mean(
                tf.square(regression_output_a - regression_output_b)
            )
            barlow_loss = self.barlow_loss(z_a, z_b)
            regression_loss = tf.keras.losses.MSE(regression_target2, regression_output_c)

            loss = (
                barlow_loss * self.loss_weight[0]
                + regression_loss * self.loss_weight[1]
                + consistency_loss * self.loss_weight[2]
            )

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def test_step(self, batch):
        y_a, y_b, regression_target = batch
        z_a = self.encoder_a(y_a, training=False)
        regression_output_a = self.regression_head(z_a, training=False)
        loss = tf.keras.losses.MSE(regression_target, regression_output_a)
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def call(self, inputs, training=False):
        encoded = self.encoder_a(inputs, training=training)
        return self.regression_head(encoded, training=training)


def build_supervised_model(init_shape=(222,), enc_sizes=(16,), reg_sizes=(1,),
                           activation='linear', conv1=1., preprocess=3,
                           learning_rate=0.005, initializer=None):
    """Build and compile a supervised (MSE-only) baseline model.

    Uses the same encoder + regression head architecture as
    ``BarlowRegressionModel`` but trains with MSE loss only on labeled data.

    Parameters
    ----------
    init_shape : tuple of int
        Input shape (default ``(222,)``).
    enc_sizes : tuple of int
        Encoder hidden layer sizes.
    reg_sizes : tuple of int
        Regression head layer sizes.
    activation : str
        Encoder Dense activation (default 'linear').
    conv1 : float
        Enable trainable Conv1D if > 0 (default 1.0).
    preprocess : int
        Savitzky-Golay derivative order (default 3).
    learning_rate : float
        Adam learning rate (default 0.005).
    initializer : keras initializer or None
        Defaults to ``glorot_uniform(seed=123)``.

    Returns
    -------
    model : keras.Model
        Compiled functional Keras model.
    """
    if initializer is None:
        initializer = tf.keras.initializers.glorot_uniform(seed=123)

    inputs = keras.Input(shape=init_shape)
    x = build_encoder(
        init_shape=init_shape, layer_sizes=enc_sizes, activation=activation,
        conv1=conv1, preprocess=preprocess, initializer=initializer,
    )(inputs)
    x = get_regression_head(
        encoder_output_size=enc_sizes, reg_sizes=reg_sizes, activation='linear',
        initializer=initializer,
    )(x)
    model = keras.Model(inputs, x)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model
