import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import Tensor
from typing import Union, List, Optional
from smoothdoq.layers import ResBlock1D


activations = tf.keras.activations
layers = tf.keras.layers
models = tf.keras.models
sequence = tf.keras.preprocessing.sequence


class BinDenoiser(models.Model):
    def __init__(
        self,
        nblocks: int = 1,
        kernel_size: Union[int, List[int]] = 1,
        filters: Union[int, List[int]] = 32,
        dilation_rate: Union[int, List[int]] = 1,
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
    eps = 1e-12
    x = x + eps
    C = tf.reduce_max(x * mask, axis, keepdims=True)
    S = tf.math.exp(x - C) * mask
    S = S / tf.reduce_sum(S, axis, keepdims=True)
    return S


def masked_add_to_one(x: Tensor, mask: Tensor, axis: int = -1) -> Tensor:
    eps = 1e-12
    x = x + eps
    S = tf.reduce_sum(x * mask, axis, keepdims=True)
    return (x * mask) / S


def compute_mask(x: List[List], maxlen: Optional[int] = None) -> Tensor:
    pad = sequence.pad_sequences(
        x, value=-1.0, dtype="float", padding="post", maxlen=maxlen
    )
    return tf.cast(pad != -1, tf.float32)


if __name__ == "__main__":

    nblocks = 2
    # kernel_size = [7] + [7] * nblocks
    kernel_size = [21, 3, 7]  # + [7] * nblocks
    filters = 128
    dilation_rate = [1, 8, 1] #+ [1, 2, 4, 8, 1, 1, 1, 1]


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
