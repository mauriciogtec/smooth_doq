import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import Tensor
from typing import Any


layers = tf.keras.layers
models = tf.keras.models


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
        norm: str = 'none',
    ) -> None:

        super(ResBlock1D, self).__init__()
        if activation == "elu":
            self.act = tf.nn.elu
        elif activation == "relu":
            self.act = tf.nn.relu
        elif activation == "leaky_relu":
            self.act = tf.nn.leaky_relu
        else:
            raise NotImplementedError(activation)
        
        if norm == "ln":
            self.norm_fn = tf.keras.layers.LayerNormalization
        elif norm == "in":
            self.norm_fn = tfa.layers.InstanceNormalization
        elif norm == "bn":
            self.norm_fn = tf.keras.layers.BatchNormalization
        elif norm == 'none':
            self.norm_fn = None
            self.use_bias = True
        else:
            raise NotImplementedError(norm)
    

        self.use_bias = use_bias
        self.conv_skip = True if strides > 1 else conv_skip
        if self.conv_skip:
            self.shortcut_ln = tf.keras.layers.LayerNormalization()
            self.shortcut_conv = layers.Conv1D(
                filters, 1, padding="same", use_bias=use_bias, strides=strides
            )

        # block 1
        self.ln1 = self.norm_fn()
        self.conv1 = layers.Conv1D(
            filters // 4, 1,
            padding="same",
            use_bias=use_bias,
            strides=1
        )

        # block 2
        self.ln2 = self.norm_fn()
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
        self.ln3 = self.norm_fn()
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
                shortcut = self.shortcut_ln(shortcut)
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
        activation: str = "elu",
        dilation_rate: int = 1,
        strides: int = 1,
        conv_skip: bool = False,
        use_bias: bool = False,
        norm: str = 'none',
    ) -> None:

        super(ResBlock1D_2, self).__init__()
        if activation == "elu":
            self.act = tf.nn.elu
        elif activation == "relu":
            self.act = tf.nn.relu
        elif activation == "leaky_relu":
            self.act = tf.nn.leaky_relu
        else:
            raise NotImplementedError(activation)

        if norm == "ln":
            self.norm_fn = tf.keras.layers.LayerNormalization
        elif norm == "in":
            self.norm_fn = tfa.layers.InstanceNormalization
        elif norm == "bn":
            self.norm_fn = tf.keras.layers.BatchNormalization
        elif norm == 'none':
            self.norm_fn = None
        else:
            raise NotImplementedError(norm)

        self.use_bias = use_bias
        self.conv_skip = True if strides > 1 else conv_skip
        if self.conv_skip:
            self.shortcut_ln = tf.keras.layers.LayerNormalization()
            self.shortcut_conv = layers.Conv1D(
                filters, 1, padding="same", use_bias=use_bias, strides=strides
            )

        # block 2
        self.ln1 = self.norm_fn()
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

        # block 3
        self.ln2 = self.norm_fn()
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
                shortcut = self.shortcut_ln(shortcut)
            shortcut = self.act(shortcut)
            shortcut = self.shortcut_conv(shortcut)

        return shortcut + x


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
