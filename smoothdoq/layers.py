import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import Tensor
from typing import Any, Callable


layers = tf.keras.layers
models = tf.keras.models
initializers = tf.keras.initializers


def get_act_fn(activation: str) -> Callable:
    if activation == "none":
        act = None
    elif activation == "elu":
        act = tf.nn.elu
    elif activation == "relu":
        act = tf.nn.relu
    elif activation == "leaky_relu":
        act = tf.nn.leaky_relu
    elif activation == "gelu":
        act = tf.nn.gelu
    else:
        raise NotImplementedError(activation)
    return act


def get_norm_layer(norm: str) -> layers.Layer:
    if norm == "none":
        norm_fn = None
    elif norm == "ln":
        norm_fn = tf.keras.layers.LayerNormalization()
    elif norm == "in":
        norm_fn = tfa.layers.InstanceNormalization()
    elif norm == "bn":
        norm_fn = tf.keras.layers.BatchNormalization()
    elif norm == "none":
        norm_fn = None
    else:
        raise NotImplementedError(norm)
    return norm_fn


class ResBlock1D(models.Model):
    """Defines a residual block which takes as input
    and output a 3d tensor of same dimensions
    batch x values x channels"""

    def __init__(
        self,
        kernel_size: int,
        filters: int,
        activation: str = "elu",
        dilation_rate: int = 1,
        strides: int = 1,
        conv_skip: bool = False,
        use_bias: bool = False,
        norm: str = "none",
    ) -> None:

        super(ResBlock1D, self).__init__()

        self.act = get_act_fn(activation)
        self.norm_fn = get_norm_layer(norm)
        self.use_bias = (norm == 'none')
        self.conv_skip = True if strides > 1 else conv_skip
        if self.conv_skip:
            self.shortcut_ln = tf.keras.layers.LayerNormalization()
            self.shortcut_conv = layers.Conv1D(
                filters, 1, padding="same", use_bias=use_bias, strides=strides
            )

        # block 1
        self.ln1 = self.norm_fn
        self.conv1 = layers.Conv1D(
            filters // 4, 1, padding="same", use_bias=use_bias, strides=1
        )

        # block 2
        self.ln2 = self.norm_fn
        self.conv2 = []
        for i in range(4):
            self.conv2.append(
                tf.keras.layers.Conv1D(
                    filters // 4,
                    kernel_size + i * 2,
                    dilation_rate=dilation_rate,
                    padding="same",
                    use_bias=use_bias,
                )
            )

        # block 3
        self.ln3 = self.norm_fn
        self.conv3 = tf.keras.layers.Conv1D(
            filters, 1, padding="same", use_bias=use_bias, strides=strides
        )

    def call(self, inputs, training=None) -> Tensor:
        x = inputs
        if not self.use_bias:
            x = self.ln1(x, training=training)
        x = self.act(x)
        x = self.conv1(x)
        if not self.use_bias:
            x = self.ln2(x, training=training)
        x = self.act(x)
        x = tf.concat([layer(x) for layer in self.conv2], -1)
        if not self.use_bias:
            x = self.ln3(x, training=training)
        x = self.act(x)
        x = self.conv3(x)

        shortcut = inputs
        if self.conv_skip:
            if not self.use_bias:
                shortcut = self.shortcut_ln(shortcut, training=training)
            shortcut = self.act(shortcut)
            shortcut = self.shortcut_conv(shortcut)

        return shortcut + x


class ResBlock1D_2(models.Model):
    """Defines a residual block which takes as input
    and output a 3d tensor of same dimensions
    batch x values x channels"""

    def __init__(
        self,
        kernel_size: int,
        filters: int,
        activation: str = "gelu",
        dilation_rate: int = 1,
        strides: int = 1,
        conv_skip: bool = False,
        use_bias: bool = False,
        norm: str = "none",
        residual: bool = True,
    ) -> None:

        super(ResBlock1D_2, self).__init__()

        self.act = get_act_fn(activation)
        self.residual = residual
        self.use_bias = (norm == 'none')
        self.conv_skip = True if strides > 1 else conv_skip
        if self.conv_skip:
            self.shortcut_ln = tf.keras.layers.LayerNormalization()
            self.shortcut_conv = layers.Conv1D(
                filters, 1, padding="same", use_bias=use_bias, strides=strides
            )

        self.ln1 = self.norm_fn(norm)
        self.conv1 = []
        for i in range(4):
            self.conv1.append(
                tf.keras.layers.Conv1D(
                    filters // 4,
                    kernel_size + i * 2,
                    dilation_rate=dilation_rate,
                    padding="same",
                    use_bias=use_bias,
                )
            )

        self.ln2 = self.norm_fn(norm)
        self.conv2 = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
        )

    def call(self, inputs, training=None) -> Tensor:
        x = inputs
        if not self.use_bias:
            x = self.ln1(x, training=training)
        x = self.act(x)
        x = tf.concat([layer(x) for layer in self.conv1], -1)
        if not self.use_bias:
            x = self.ln2(x, training=training)
        x = self.act(x)
        x = self.conv2(x)

        shortcut = inputs
        if self.conv_skip:
            if not self.use_bias:
                shortcut = self.shortcut_ln(shortcut, training=training)
            shortcut = self.act(shortcut)
            shortcut = self.shortcut_conv(shortcut)

        if self.residual:
            x = x + shortcut

        return x


def adain1D(x, weights) -> Tensor:
    # instance statistics
    mu = tf.reduce_mean(x, 1, keepdims=True)
    sig = 1e-3 + tf.math.reduce_std(x, 1, keepdims=True)

    # normalize
    gam, beta = tf.split(tf.expand_dims(weights, 1), 2, -1)
    z = gam * (x - mu) / sig + beta
    return z


class AdaInConvBlock1D(models.Model):
    """Defines a residual block which takes as input
    and output a 3d tensor of batch x values x channels"""

    def __init__(
        self,
        kernel_size: int,
        filters: int,
        activation: str = "gelu",
        dilation_rate: int = 1,
        strides: int = 1,
        norm: str = "none",
        residual: bool = True,
        upsample: bool = True
    ) -> None:

        super(AdaInConvBlock1D, self).__init__()

        self.act = get_act_fn(activation)
        self.adain = (norm == 'adain')
        self.use_bias = (norm == 'none')
        if self.adain:
            self.norm_fn = None
        else:
            self.norm_fn = get_norm_layer(norm)
        self.residual = residual
        self.strides = strides
        self.upsample = upsample
        self.strides = strides
        conv_strides = strides if not upsample else 1
        # initializer = initializers.TruncatedNormal(mean=0., stddev=0.5)
        self.conv = layers.Conv1D(
            filters,
            kernel_size,
            strides=conv_strides,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=self.use_bias,
            # kernel_initializer=initializer
        )
        self.upsample = layers.UpSampling1D(strides) if upsample else None

    def call(self, inputs, weights=None, training=None) -> Tensor:
        x = inputs
        x = self.act(x)
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.conv(x)
        if weights is not None:
            x = adain1D(x, weights)
        elif self.norm_fn is not None:
            x = self.norm_fn(x, training=training)
        if self.residual:
            x = x + inputs

        return x


# class TransposeResBlock1D_2(models.Model):
#     """Defines a tranpose (upsample) residual block
#     which takes as input
#     and output a 3d tensor of same dimensions
#     batch x values x channels"""

#     def __init__(
#         self,
#         kernel_size: int,
#         filters: int,
#         activation: str = "gelu",
#         dilation_rate: int = 1,
#         strides: int = 1,
#         conv_skip: bool = False,
#         use_bias: bool = False,
#         norm: str = "none",
#         residual: bool = True,
#     ) -> None:

#         super(TransposeResBlock1D_2, self).__init__()
#         self.act = get_act_fn(activation)
#         self.norm_fn = get_norm_layer(norm)
#         self.residual = residual
#         self.use_bias = use_bias
#         self.conv_skip = True if strides > 1 else conv_skip
#         if self.conv_skip:
#             self.shortcut_ln = tf.keras.layers.LayerNormalization()
#             self.shortcut_conv = layers.Conv1D(
#                 filters, 1, padding="same", use_bias=use_bias, strides=strides
#             )

#         self.ln1 = self.norm_fn()
#         self.conv1 = []
#         for i in range(4):
#             self.conv1.append(
#                 tf.keras.layers.Conv1DTranspose(
#                     filters // 4,
#                     kernel_size + i * 2,
#                     dilation_rate=dilation_rate,
#                     padding="same",
#                     use_bias=use_bias,
#                 )
#             )

#         self.ln2 = self.norm_fn()
#         self.conv2 = tf.keras.layers.Conv1DTranspose(
#             filters,
#             kernel_size,
#             dilation_rate=dilation_rate,
#             padding="same",
#             use_bias=use_bias,
#         )

#     def call(self, inputs, training=None) -> Tensor:
#         x = inputs
#         if not self.use_bias:
#             x = self.ln1(x, training=training)
#         x = self.act(x)
#         x = tf.concat([layer(x) for layer in self.conv1], -1)
#         if not self.use_bias:
#             x = self.ln2(x, training=training)
#         x = self.act(x)
#         x = self.conv2(x)

#         shortcut = inputs
#         if self.conv_skip:
#             if not self.use_bias:
#                 shortcut = self.shortcut_ln(shortcut, training=training)
#             shortcut = self.act(shortcut)
#             shortcut = self.shortcut_conv(shortcut)

#         if self.residual:
#             x = x + shortcut

#         return x


class LinearBlock(models.Model):
    """Defines a linear block (dense + activation)"""

    def __init__(
        self, units: int, norm='none', activation="relu"
    ) -> None:

        super(LinearBlock, self).__init__()
        self.act = get_act_fn(activation)
        self.norm_fn = get_norm_layer(norm)
        self.use_bias = (norm == 'none')
        self.dense = layers.Dense(units, use_bias=self.use_bias)

    def call(self, inputs, training=None) -> Tensor:
        x = inputs
        if self.norm_fn is not None:
            x = self.norm_fn(x, training=training)
        if self.act is not None:
            x = self.act(x)
        x = self.dense(x)
        return x


# class AdaInRes1DBlock(models.Model):
#     """Defines a linear block (dense + activation)"""

#     def __init__(
#         self,
#         kernel_size: int,
#         filters: int,
#         activation: str = "gelu",
#         dilation_rate: int = 1,
#         residual: bool = True,
#     ) -> None:

#         super(AdaInRes1DBlock, self).__init__()
#         self.act = get_act_fn(activation)

#         self.conv1 = []
#         if filters > 4:
#             for i in range(4):
#                 self.conv1.append(
#                     TransposeResBlock1D_2(
#                         filters // 4,
#                         kernel_size + i * 2,
#                         dilation_rate=dilation_rate,
#                         padding="same",
#                         use_bias=False,
#                     )
#                 )

#     def call(self, inputs, adain_weights, training=None) -> Tensor:
#         x = inputs
#         x = tf.concat([layer(x) for layer in self.conv1], -1)

#         # instance statistics
#         mu = tf.reduce_mean(inputs, -1, keepdims=True)
#         sig = 1e-3 + tf.math.reduce_std(inputs, -1, keepdims=True)

#         # normalize
#         gam, beta = tf.split(tf.expand_dims(adain_weights, -1), 2, 1)
#         z = gam * (x - mu) / sig + beta
#         out = self.act(z) + inputs

#         return out


class Ensemble(models.Model):
    """Takes a list of models and calls them in a loop
    and concatenates the outputs"""

    def __init__(self, models: list, stack_dim: int = 0) -> None:
        super(Ensemble, self).__init__()
        self.models = models
        self.stack_dim = stack_dim

    def call(self, *args, **kwargs) -> Any:
        outputs = [m(*args, **kwargs) for m in self.models]
        return tf.stack(outputs, self.stack_dim)

    def call_at(self, i: int, *args, **kwargs):
        return self.models[i](*args, **kwargs)


class SoftGlobalMaxPool1D(layers.Layer):
    """Global pool layer"""

    def __init__(self, alpha: float = 1.0) -> None:
        super(SoftGlobalMaxPool1D, self).__init__()
        self.alpha = alpha

    def call(self, x: Tensor) -> Tensor:
        x = tf.reduce_logsumexp(self.alpha * x, 1)
        return x
