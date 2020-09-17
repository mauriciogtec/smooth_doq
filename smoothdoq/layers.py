import tensorflow as tf
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
        **kwargs
    ) -> None:

        super(ResBlock1D, self).__init__(**kwargs)
        self.act = layers.Activation(activation)
        self.conv_skip = True if strides > 1 else conv_skip
        if self.conv_skip:
            self.shortcut_ln = tf.keras.layers.LayerNormalization()
            self.shortcut_conv = layers.Conv1D(
                filters, 1, padding="same", use_bias=False, strides=strides
            )

        # block 1
        # self.ln1 = MaskedLayerNorm()
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.conv1 = layers.Conv1D(
            filters // 4, 1,
            padding="same",
            use_bias=True,
            strides=1
        )

        # block 2
        # self.ln2 = MaskedLayerNorm()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.conv2 = []
        for i in range(4):
            self.conv2.append(
                tf.keras.layers.Conv1D(
                    filters // 4,
                    kernel_size + i * 2,
                    dilation_rate=dilation_rate,
                    padding="same",
                    use_bias=False,
                )
            )

        # block 3
        # self.ln3 = MaskedLayerNorm()
        self.ln3 = tf.keras.layers.LayerNormalization()
        self.conv3 = tf.keras.layers.Conv1D(
            filters, 1, padding="same", use_bias=False, strides=strides
        )

    def call(self, inputs, training=None) -> Tensor:
        x = inputs
        x = self.ln1(x, training=training)
        x = self.act(x)
        x = self.conv1(x)
        x = self.ln2(x, training=training)
        x = self.act(x)
        x = tf.concat([layer(x) for layer in self.conv2], -1)
        x = self.ln3(x, training=training)
        x = self.act(x)
        x = self.conv3(x)

        shortcut = inputs
        if self.conv_skip:
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
