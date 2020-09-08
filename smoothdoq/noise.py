import numpy as np
from smoothdoq.binned import BinnedDistribution
from numpy import ndarray

# list of noise types
# 1. rounding noise
# 2. multiplicative over disperssion
# 3. negative binomial disperssion
# noise types are just subclasses of distributions


class RoundingNoise(BinnedDistribution):
    """Adds rounding noise. It makes sure the support is not changed
    by rounding values towards the end of the interval."""

    def __init__(self, distribution: BinnedDistribution, stride: int) -> None:
        assert 1 < stride < 2 * distribution.n_bins + 1
        super().__init__(distribution.n_bins)
        self.distribution = distribution
        self.stride = stride
        self.batch_size = self.distribution.batch_size

    def target_bins(self) -> list:
        target_bin = []
        w = self.stride
        for i in range(self.n_bins):
            if i <= w // 2:
                target_bin.append(0)
            elif i >= self.n_bins - w // 2:
                target_bin.append(self.n_bins - 1)
            else:
                target_bin.append(w * int(round((i - w // 2) / w)))
        return target_bin

    def sample(self, size: int = 1, offset=None) -> ndarray:
        N = self.n_bins
        tmp = self.distribution.sample(size)
        targets = self.target_bins()
        out = np.zeros_like(tmp)
        for i in range(self.batch_size):
            for j in range(self.n_bins):
                out[i, targets[j]] += tmp[i, j]
        return out


class MultiplicativeDispersion(BinnedDistribution):
    """Adds rounding noise. It makes sure the support is not changed
    by rounding values towards the end of the interval."""

    def __init__(self, distribution: BinnedDistribution, sigma: float) -> None:
        assert sigma > 0.0
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
        noise_ratio: float,
        disp_coef: float,
    ) -> None:
        assert 0.0 < noise_ratio < 1.0
        assert disp_coef > 0.0
        super().__init__(distribution.n_bins)
        self.distribution = distribution
        self.disp_coef = disp_coef
        self.noise_ratio = noise_ratio
        self.batch_size = self.distribution.batch_size
        self.pdf = self.distribution.pdf

    def sample(self, size: int = 1, adjust_size=True) -> ndarray:
        N = self.n_bins
        signal_size = np.random.binomial(size, 1.0 - self.noise_ratio)
        signal = self.distribution.sample(signal_size)
        mu = (size - signal_size) / self.n_bins

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
            p = np.expand_dims(r / (r + 2 * mu), -1)
            background = np.random.negative_binomial(
                r, p, (self.batch_size, self.n_bins)
            )
            for i in range(self.batch_size):
                diff = (size - signal_size) - background[i].sum()
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
