import tensorflow as tf
from tensorflow import Tensor
from typing import List, Optional

# from smoothdoq.layers import ResBlock1D_2 as ResBlock
# from smoothdoq.layers import TransposeResBlock1D_2 as ResBlockTranspose
# from smoothdoq.layers import LinearBlock, AdaInRes1DBlock
from smoothdoq.layers import AdaInConvBlock1D, LinearBlock, SoftGlobalMaxPool1D


layers = tf.keras.layers
models = tf.keras.models


class ConvPipe(models.Model):
    def __init__(
        self,
        kernel_size: List[int] = [3],
        filters: List[int] = [4],
        strides: List[int] = [1],
        dilation: List[int] = [1],
        activation: str = "gelu",
        norm: str = "in",
        residual: List[bool] = [True],
        pool: bool = False,
        pool_alpha: float = 1.0,
        upsample: bool = False,
    ) -> None:
        super(ConvPipe, self).__init__()
        self.filters = filters
        self.blocks = []
        for k, f, s, r, d in zip(
            kernel_size, filters, strides, residual, dilation
        ):
            blk = AdaInConvBlock1D(
                kernel_size=k,
                filters=f,
                activation=activation,
                dilation_rate=d,
                norm=norm,
                strides=s,
                residual=r,
                upsample=upsample,
            )
            self.blocks.append(blk)
        self.pool = SoftGlobalMaxPool1D(alpha=pool_alpha) if pool else None

    def call(
        self,
        inputs: Tensor,
        w: Optional[Tensor] = None,
        training: Optional[bool] = True,
    ) -> Tensor:
        x = inputs
        if w is not None:
            weights = tf.split(w, [2 * f for f in self.filters], -1)
        else:
            weights = [None] * len(self.filters)

        for blk, w in zip(self.blocks, weights):
            x = blk(x, w)

        if self.pool is not None:
            x = self.pool(x)
        return x


# class Decoder(models.Model):
#     def __init__(
#         self,
#         kernel_size: List[int] = [3],
#         filters: List[int] = [4],
#         strides: List[int] = [1],
#         activation: str = "gelu",
#         norm: str = "adain",
#         residual: List[bool] = [True],
#     ) -> None:
#         super(Encoder, self).__init__()
#         self.kernel_size = kernel_size
#         self.filters = filters
#         self.strides = strides
#         self.activation = activation
#         self.norm = norm
#         self.residual = residual

#         blocks = []
#         for k, f, s, r in zip(kernel_size, filters, strides, residual):
#             blk = AdaInConvBlock1D(
#                 kernel_size=k,
#                 filters=f,
#                 activation=activation,
#                 norm=norm,
#                 strides=s,
#                 residual=r,
#                 transpose=True,
#             )
#             blocks.append(blk)
#         self._model = tf.keras.Sequential(blocks)

#     def call(self, x: Tensor, w: Tensor) -> Tensor:
#         return self._model(x, w)


class MLP(models.Model):
    def __init__(
        self, output_dim, inner_dim, n_blk, norm="relu", activation="relu"
    ) -> None:
        super(MLP, self).__init__()

        blocks = []
        for i in range(n_blk - 1):
            blocks += [
                LinearBlock(inner_dim, norm=norm, activation=activation)
            ]
        blocks += [LinearBlock(output_dim, "none", "none")]
        self.model = tf.keras.Sequential(blocks)

    def call(self, x: Tensor) -> Tensor:
        return self.model(x)


# class AdaInResPipe(models.Model):
#     """AdaIn pipe of residual blocks.
#     Similar to encoder but takes as input the adain
#     normalization weights"""

#     def __init__(
#         self,
#         kernel_size: List[int] = [3],
#         filters: List[int] = [4],
#         activation: str = "gelu",
#         residual: List[bool] = [True],
#     ) -> None:
#         super(Encoder, self).__init__()
#         self.kernel_size = kernel_size
#         self.filters = filters
#         self.activation = activation
#         self.residual = residual

#         self.blocks = []
#         self.n_blk = len(kernel_size)
#         for k, f, s, r in zip(kernel_size, filters):
#             blk = AdaInRes1DBlock(
#                 kernel_size=k, filters=f, activation=activation, residual=r,
#             )
#             self.blocks.append(blk)

#     def call(self, inputs: Tensor, weights: Tensor) -> Tensor:
#         x = inputs
#         weights = tf.split(weights, self.n_blk)
#         for blk, w in zip(self.blocks, weights):
#             x = blk(x, w)
#         return x
