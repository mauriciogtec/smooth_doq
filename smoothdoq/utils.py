import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import layers, initializers, regularizers, constraints


class MaskedLayerNorm(layers.Layer):
    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=1e-14,
        gamma_initializer="ones",
        beta_initializer="zeros",
        gamma_regularizer=None,
        beta_regularizer=None,
        gamma_constraint=None,
        beta_constraint=None,
        **kwargs
    ):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(MaskedLayerNorm, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name="gamma",
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name="beta",
            )
        super(MaskedLayerNorm, self).build(input_shape)

    def call(self, x, dim=-1, N=None, training=None):
        # if N is none then it is jsut the number of cols
        if N is None:
            N = tf.cast(tf.shape(x), tf.float32)[dim]

        # prepare for broadcasting
        if len(N.shape) == 1:
            batch_size = x.shape[0]
            N = tf.reshape(N, (batch_size, 1, 1))

        mean = tf.reduce_sum(x, dim, keepdims=True) / N
        sumsq = tf.reduce_sum((x - mean) ** 2, dim, keepdims=True)
        std = tf.math.sqrt(sumsq / (N - 1.0)) + self.epsilon
        outputs = (x - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


def set_seed_everywhere(seed: int) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
