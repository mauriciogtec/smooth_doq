import numpy as np
from smoothdoq.binned import BinnedDistribution
from numpy import ndarray
from typing import Union

# list of noise types
# 1. rounding noise
# 2. multiplicative over disperssion
# 3. negative binomial disperssion
# noise types are just subclasses of distributions


class RoundingNoise(BinnedDistribution):
    """Adds rounding noise. It makes sure the support is not changed
    by rounding values towards the end of the interval."""

    def __init__(
        self, distribution: BinnedDistribution, stride: Union[int, ndarray]
    ) -> None:
        # assert 1 < stride < 2 * distribution.n_bins + 1
        super().__init__(distribution.n_bins)
        self.distribution = distribution
        if isinstance(stride, int):
            self.stride = [stride] * self.distribution.batch_size
        else:
            self.stride = stride
        self.batch_size = self.distribution.batch_size

    def target_bins(self, w: int) -> list:
        target_bin = []
        # if not isinstance(w, list):
        #     stride = [stride] * ?
        for i in range(self.n_bins):
            if i <= w // 2:
                target_bin.append(0)
            elif i >= self.n_bins - w // 2:
                target_bin.append(self.n_bins - 1)
            else:
                target_bin.append(w * int(round((i - w // 2) / w)))
        return target_bin

    def sample(self, size: int = 1, offset=None) -> ndarray:
        # N = self.n_bins
        tmp = self.distribution.sample(size)
        out = np.zeros_like(tmp)
        for i in range(self.batch_size):
            targets = self.target_bins(self.stride[i])
            for j in range(self.n_bins):
                out[i, targets[j]] += tmp[i, j]
        return out


class MultiplicativeDispersion(BinnedDistribution):
    """Adds rounding noise. It makes sure the support is not changed
    by rounding values towards the end of the interval."""

    def __init__(
        self, distribution: BinnedDistribution, sigma: Union[ndarray, float]
    ) -> None:
        # assert sigma > 0.0
        super().__init__(distribution.n_bins)
        self.distribution = distribution
        self.sigma = sigma
        self.batch_size = self.distribution.batch_size
        self.pdf = self.distribution.pdf

    def dispersions(self) -> list:
        w = np.random.normal(0, self.sigma, (self.batch_size, self.n_bins))
        w = np.exp(w)
        w /= w.mean(-1, keepdims=True)
        return w

    def sample(self, size: int = 1, adjust_size=True) -> ndarray:
        weights = self.dispersions()
        out = np.ceil(self.distribution.sample(size) * weights).astype(int)

        # tiny adjustment to guarantee it returns the correct number
        # adjust extra by randomly adding removing proportional to a bin
        if adjust_size:
            for i in range(self.batch_size):
                diff = size - out[i].sum()
                sign = 1 if diff > 0 else -1
                while diff != 0:
                    p = (out[i] + 1e-14) / (out[i].sum() + 1e-14)
                    targets = np.random.choice(
                        range(self.n_bins), p=p, size=diff * sign
                    )
                    for j in targets:
                        if out[i, j] + sign >= 0:
                            out[i, j] += sign
                            diff -= sign

        return out


class NegBinBackgroundNoise(BinnedDistribution):
    """Adds rounding noise. It makes sure the support is not changed
    by rounding values towards the end of the interval."""

    def __init__(
        self,
        distribution: BinnedDistribution,
        noise_ratio: Union[float, ndarray],
        disp_coef: Union[float, ndarray],
    ) -> None:
        # assert 0.0 < noise_ratio < 1.0
        # assert disp_coef > 0.0
        super().__init__(distribution.n_bins)
        self.distribution = distribution
        self.disp_coef = disp_coef
        self.noise_ratio = noise_ratio
        self.batch_size = self.distribution.batch_size
        self.pdf = self.distribution.pdf

    def sample(self, size: int = 1, adjust_size=True) -> ndarray:
        # N = self.n_bins
        sr = self.noise_ratio
        if isinstance(sr, float):
            sr = [sr] * self.distribution.batch_size
        signal = []
        total = self.distribution.sample(size)
        signal = np.zeros_like(total)
        for j, s in enumerate(sr):
            signal[j] = np.random.binomial(total[j], s)
        mu = (total - signal) / self.n_bins

        if not adjust_size:
            r = self.disp_coef
            p = np.expand_dims(r / (r + mu), -1)
            background = np.random.negative_binomial(
                r, p, (self.batch_size, self.n_bins)
            )
        else:
            r = self.disp_coef
            # making mu larger implies that high probability will
            # oversample and then slim down
            p = r / (r + 2 * mu)
            if len(p.shape) == 1:
                p = np.expand_dims(p, -1)
            background = np.random.negative_binomial(
                r, p, (self.batch_size, self.n_bins)
            )
            for i in range(self.batch_size):
                diff = (total[i] - signal[i]).sum() - background[i].sum()
                sign = 1 if diff > 0 else -1
                while diff != 0:
                    p = background[i] + 1e-14
                    p /= p.sum()
                    targets = np.random.choice(
                        range(self.n_bins), p=p, size=diff * sign
                    )
                    for j in targets:
                        if background[i, j] + sign >= 0:
                            background[i, j] += sign
                            diff -= sign

        return signal + background
