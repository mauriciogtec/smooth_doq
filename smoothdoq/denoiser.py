import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import activations, layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# local
from smoothdoq.utils import MaskedLayerNorm

# typing
from tensorflow import Tensor
from typing import Union, List, Optional


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
        **kwargs
    ) -> None:
        super(ResBlock1D, self).__init__(**kwargs)

        self.act = layers.Activation(activation)

        # block 1
        # self.ln1 = MaskedLayerNorm()
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.conv1 = layers.Conv1D(
            filters // 4, 1, padding="same", use_bias=False,
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
            filters, 1, padding="same", use_bias=False,
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
        return inputs + x


class BinDenoiser(models.Model):
    def __init__(
        self,
        nblocks: int = 1,
        kernel_size: Union[int, List[int]] = 1,
        filters: Union[int, List[int]] = 32,
        dilation_rate: Union[int, List[int]] = 32,
        alpha: float = 0.5,
        activation: str = "elu",
        dropout: float = 0.0,
        **kwargs
    ) -> None:
        super(BinDenoiser, self).__init__(**kwargs)

        self.nblocks = nblocks
        kernel_size = (
            [kernel_size] * (nblocks + 1)
            if isinstance(kernel_size, int)
            else kernel_size
        )
        filters = (
            [filters] * (nblocks + 1) if isinstance(filters, int) else filters
        )
        dilation_rate = (
            [dilation_rate] * (nblocks + 1)
            if isinstance(dilation_rate, int)
            else dilation_rate
        )
        self.dropout = dropout

        self.blocks = []

        self.conv0_list = []
        for i in range(4):
            layer = layers.Conv1D(
                filters=filters[0] // 4,
                kernel_size=kernel_size[0] + i * 4,
                dilation_rate=dilation_rate[0],
                padding="same",
                use_bias=False,
            )
            self.conv0_list.append(layer)
        for k, f, d in zip(kernel_size[1:], filters[1:], dilation_rate[1:]):
            new_block = ResBlock1D(
                kernel_size=k,
                filters=f,
                dilation_rate=d,
                activation=activation,
            )
            self.blocks.append(new_block)
        self.conv_final = layers.Conv1D(
            filters=1, kernel_size=1, padding="same"
        )

    def call(
        self,
        inputs: Tensor,
        mask: Optional[Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tensor:
        # N = None if mask is None else tf.reduce_sum(mask, 1)
        x = tf.expand_dims(inputs, -1)
        x = tf.concat([layer(x) for layer in self.conv0_list], -1)
        for block in self.blocks:
            # N = None
            x = block(x, training=training)
        if training and self.dropout > 0.0:
            R = tf.random.uniform((x.shape[0], 1, x.shape[2]))
            R = tf.cast(R < self.dropout, tf.float32)
            x = x * R
        x = self.conv_final(x)
        x = tf.squeeze(x, -1)

        return x


def masked_softmax(x: Tensor, mask: Tensor, axis: int = -1) -> Tensor:
    eps = 1e-14
    C = tf.reduce_max(x * mask, axis, keepdims=True)
    S = (tf.math.exp(x - C) + eps) * mask
    S = S / (tf.reduce_sum(S, axis, keepdims=True) + eps)
    return S


def compute_mask(x: List[List], maxlen: Optional[int] = None) -> Tensor:
    pad = pad_sequences(
        x, value=-1.0, dtype="float", padding="post", maxlen=maxlen
    )
    return tf.cast(pad != -1, tf.float32)


if __name__ == "__main__":

    nblocks = 5
    kernel_size = [5, 3, 3, 3, 3]
    filters = 16
    dilation_rate = [1, 1, 2, 4, 8]

    model = BinDenoiser(
        nblocks=nblocks,
        kernel_size=kernel_size,
        filters=filters,
        dilation_rate=dilation_rate,
    )

    D = [[1, -1.1, 1], [0.2, 0.5, 0.9, 1.0]]
    x = pad_sequences(D, dtype="float", padding="post")
    mask = compute_mask(D)
    y = model(x, mask=mask, training=True)
    y = masked_softmax(y, mask, -1)

    d = np.exp(-0.5 * np.linspace(-3.0, 3.0, 100) ** 2) / np.sqrt(2 * np.pi)
    d /= d.sum()
    x = tf.math.log(tf.constant(d) + 1e-14)
    x -= tf.reduce_mean(x)
    x = tf.expand_dims(x, 0)
    y = model(x)
    y = tf.math.softmax(y, -1)
    y_ = y[0].numpy()

    u = tf.math.softmax(x)[0].numpy()
    plt.plot(d, label="raw")
    plt.plot(u, label="rec")
    plt.plot(y_, label="post")
    plt.legend()
    plt.show()

    0
