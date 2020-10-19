import tensorflow as tf
from tensorflow import Tensor
from typing import Union, List, Optional
from smoothdoq.layers import ResBlock1D, ResBlock1D_2


activations = tf.keras.activations
layers = tf.keras.layers
models = tf.keras.models
sequence = tf.keras.preprocessing.sequence


class FreeFormGAN(models.Model):
    def __init__(
        self,
        nblocks: int = 1,
        kernel_size: Union[int, List[int]] = 1,
        filters: Union[int, List[int]] = 32,
        dilation_rate: Union[int, List[int]] = 32,
        normalization: bool = False,
        activation: str = "elu",
        **kwargs
    ) -> None:
        super(FreeFormGAN, self).__init__(**kwargs)

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

        self.blocks = []
        self.conv0_list = []
        for i in range(4):
            layer = layers.Conv1D(
                filters=filters[0] // 4,
                kernel_size=kernel_size[0] + i * 4,
                dilation_rate=dilation_rate[0],
                padding="same",
                use_bias=True,
            )
            self.conv0_list.append(layer)
        for k, f, d in zip(kernel_size[1:], filters[1:], dilation_rate[1:]):
            new_block = ResBlock1D(
                kernel_size=k,
                filters=f,
                dilation_rate=d,
                activation=activation,
                use_bias=not normalization
            )
            self.blocks.append(new_block)
        self.conv_final = layers.Conv1D(
            filters=1, kernel_size=1, padding="same"
        )

    def call(
        self,
        logits: Tensor,
        noise: Tensor,
        training: Optional[bool] = None,
    ) -> Tensor:
        # N = None if mask is None else tf.reduce_sum(mask, 1)
        x = tf.stack([logits, noise], -1)
        x = tf.concat([layer(x) for layer in self.conv0_list], -1)
        for block in self.blocks:
            # N = None
            x = block(x, training=training)
        x = self.conv_final(x)
        x = tf.squeeze(x, -1)

        return x


class StructGAN(models.Model):
    def __init__(
        self,
        nblocks: int = 1,
        kernel_size: Union[int, List[int]] = 1,
        noise_pre_kernel_size: int = 5,
        filters: Union[int, List[int]] = 32,
        dilation_rate: Union[int, List[int]] = 32,
        activation: str = "leaky_relu",
        normalization: bool = False,
        dropout: bool = True
    ) -> None:
        super(StructGAN, self).__init__()

        self.nblocks = nblocks
        kernel_size = (
            [kernel_size] * nblocks
            if isinstance(kernel_size, int)
            else kernel_size
        )
        filters = (
            [filters] * nblocks if isinstance(filters, int) else filters
        )
        dilation_rate = (
            [dilation_rate] * nblocks
            if isinstance(dilation_rate, int)
            else dilation_rate
        )

        self.dropout = dropout
        if not dropout:
            self.conv_final = layers.Conv1D(
                filters=1, kernel_size=1, padding="same"
            )
        self.filters_out = filters[-1]
        self.blocks = []
        # self.conv0_list = []
        for k, f, d in zip(kernel_size, filters, dilation_rate):
            new_block = ResBlock1D_2(
                kernel_size=k,
                filters=f,
                dilation_rate=d,
                activation=activation,
                use_bias=(not normalization)
            )
            self.blocks.append(new_block)
        self.noise_pre = ResBlock1D_2(
            kernel_size=noise_pre_kernel_size,
            filters=2 * filters[0],
            dilation_rate=1,
            activation="relu",
            use_bias=(not normalization)
        )

    def call(
        self,
        signal: Tensor,
        noise: Tensor,
        training: Optional[bool] = None,
    ) -> Tensor:
        noise = tf.expand_dims(noise, -1)
        noise = self.noise_pre(noise, training=training)
        n1, n2 = tf.split(noise, 2, -1)
        x = tf.expand_dims(signal, -1) * n1 + n2
        for block in self.blocks:
            x = block(x, training=training)
        if self.dropout:
            indices = tf.random.uniform(
                (signal.shape[0], ), maxval=self.filters_out, dtype=tf.int64
            )
            ch_mask = tf.expand_dims(tf.one_hot(indices, self.filters_out), 1)
            nadd = tf.reduce_sum(x * (1e-3 + (1.0 - 1e-3) * ch_mask), -1)
            # nadd = tf.reduce_sum(x, 1)
        else:
            x = self.conv_final(x)
            nadd = tf.squeeze(x, -1)

        # nmult, nadd = tf.split(self.conv_final(x), 2, -1)
        # nmult = tf.squeeze(nmult, -1)
        # nadd = self.conv_final(x)

        # return nmult, nadd
        return nadd


class MunitGAN(models.Model):
    def __init__(
        self,
        nblocks: int = 1,
        kernel_size_enc: List[int] = [1],
        filters_enc: List[int] = [32],
        kernel_size_dec: List[int] = [1],
        filters_dec: List[int] = [32],
        strides_style: List[int] = [2],
        latent_units: int = 30,
        activation: str = "leaky_relu"
    ) -> None:
        super(MunitGAN, self).__init__()

        self.filters_out = filters[-1]

        self.content_blocks = []
        self.style_blocks = []
        self.decoder_blocks = []

        for k, f, s in zip(kernel_size_enc, filters_enc, strides_style):
            new_block = ResBlock1D_2(
                kernel_size=k,
                filters=f,
                activation=activation,
                norm='in'
            )
            new_block_style = ResBlock1D_2(
                kernel_size=k,
                filters=f,
                activation=activation,
                strides=s,
                norm='none'
            )
            self.content_blocks.append(new_block)
            self.style_blocks.append(new_block_style)

        for k, f in zip(kernel_size_dec, filters_dec):
            new_block = ResBlock1D_2(
                kernel_size=k,
                filters=f,
                activation=activation,
                norm='in'
            )
            self.decoder_blocks.append(new_block)
        output_layer = tf.keras.layers.Conv1D(
            kernel_size=1, filters=1, padding='same', use_bias=True
        )
        self.decoder_blocks.append(output_layer)
        self.latent_params = tf.keras.layers.Dense(2 * latent_units)
        self.adain_weights = tf.keras.layers.Dense(filters_enc[-1])


    def call(
        self,
        signal: Tensor,
        noise: Tensor,
        training: Optional[bool] = None,
    ) -> Tensor:

        content = tf.expand_dims(signal, -1)
        for block in self.content_blocks:
            content = block(content, training=training)

        style = tf.expand_dims(signal, -1)
        for block in self.style_blocks:
            style = block(style, training=training)
        style = tf.math.reduce_max(style, -1)
        mu, sig = tf.split(self.latent_params(style))
        style = mu + tf.random.normal(sig.shape) * sig

        # compute adain weights



        # nmult, nadd = tf.split(self.conv_final(x), 2, -1)
        # nmult = tf.squeeze(nmult, -1)
        # nadd = self.conv_final(x)

        # return nmult, nadd
        return nadd, mu, sig


class Discriminator(models.Model):
    def __init__(
        self,
        nblocks: int = 1,
        kernel_size: Union[int, List[int]] = 1,
        filters: Union[int, List[int]] = 32,
        dilation_rate: Union[int, List[int]] = 32,
        strides: Union[int, List[int]] = 1,
        activation: str = "elu",
        global_pool_mean: bool = True,
        **kwargs
    ) -> None:
        super(Discriminator, self).__init__(**kwargs)

        self.nblocks = nblocks
        kernel_size = (
            [kernel_size] * (nblocks + 1)
            if isinstance(kernel_size, int)
            else kernel_size
        )
        filters = (
            [filters] * (nblocks + 1) if isinstance(filters, int) else filters
        )
        strides = (
            [strides] * (nblocks + 1) if isinstance(strides, int) else strides
        )
        dilation_rate = (
            [dilation_rate] * (nblocks + 1)
            if isinstance(dilation_rate, int)
            else dilation_rate
        )

        self.blocks = []
        self.conv0_list = []
        for i in range(4):
            layer = layers.Conv1D(
                filters=filters[0] // 4,
                kernel_size=kernel_size[0] + i * 4,
                dilation_rate=dilation_rate[0],
                strides=strides[0],
                padding="same",
                use_bias=True,
            )
            self.conv0_list.append(layer)
        for k, f, d, s in zip(
            kernel_size[1:], filters[1:], dilation_rate[1:], strides[1:]
        ):
            new_block = ResBlock1D(
                kernel_size=k,
                filters=f,
                dilation_rate=d,
                activation=activation,
                strides=s,
            )
            self.blocks.append(new_block)
        self.dense_final = layers.Dense(1)
        self.global_pool_mean = global_pool_mean

    def call(
        self,
        inputs: Tensor,
        training: Optional[bool] = None,
    ) -> Tensor:
        x = tf.expand_dims(inputs, -1)
        x = tf.concat([layer(x) for layer in self.conv0_list], -1)
        for block in self.blocks:
            x = block(x, training=training)
        if self.global_pool_mean:
            x = tf.reduce_mean(x, 1)
        else:
            x = tf.reduce_max(x, 1)
        x = self.dense_final(x)
        x = tf.squeeze(x, -1)

        return x


class Discriminator2(models.Model):
    def __init__(
        self,
        nblocks: int = 1,
        kernel_size: Union[int, List[int]] = 1,
        filters: Union[int, List[int]] = 32,
        dilation_rate: Union[int, List[int]] = 32,
        strides: Union[int, List[int]] = 1,
        activation: str = "elu",
        global_pool_mean: bool = True,
        **kwargs
    ) -> None:
        super(Discriminator, self).__init__(**kwargs)

        self.nblocks = nblocks
        kernel_size = (
            [kernel_size] * (nblocks + 1)
            if isinstance(kernel_size, int)
            else kernel_size
        )
        filters = (
            [filters] * (nblocks + 1) if isinstance(filters, int) else filters
        )
        strides = (
            [strides] * (nblocks + 1) if isinstance(strides, int) else strides
        )
        dilation_rate = (
            [dilation_rate] * (nblocks + 1)
            if isinstance(dilation_rate, int)
            else dilation_rate
        )

        self.blocks = []
        self.conv0_list = []
        for i in range(4):
            layer = layers.Conv1D(
                filters=filters[0] // 4,
                kernel_size=kernel_size[0] + i * 4,
                dilation_rate=dilation_rate[0],
                strides=strides[0],
                padding="same",
                use_bias=True,
            )
            self.conv0_list.append(layer)
        for k, f, d, s in zip(
            kernel_size[1:], filters[1:], dilation_rate[1:], strides[1:]
        ):
            new_block = ResBlock1D(
                kernel_size=k,
                filters=f,
                dilation_rate=d,
                activation=activation,
                strides=s,
            )
            self.blocks.append(new_block)
        self.dense_final = layers.Dense(1)
        self.global_pool_mean = global_pool_mean

    def call(
        self,
        inputs: Tensor,
        training: Optional[bool] = None,
    ) -> Tensor:
        x = tf.expand_dims(inputs, -1)
        x = tf.concat([layer(x) for layer in self.conv0_list], -1)
        for block in self.blocks:
            x = block(x, training=training)
        if self.global_pool_mean:
            x = tf.reduce_mean(x, 1)
        else:
            x = tf.reduce_max(x, 1)
        x = self.dense_final(x)
        x = tf.squeeze(x, -1)

        return x


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

