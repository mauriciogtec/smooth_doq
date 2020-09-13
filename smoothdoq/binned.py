from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from numpy import ndarray
from typing import List


class BinnedDistribution(ABC):
    """Abstract class for binned distributions"""

    def __init__(self, n_bins: int) -> None:
        assert n_bins >= 1
        self.n_bins = n_bins
        super().__init__()

    @abstractmethod
    def sample(self, size: int = 1) -> ndarray:
        """Returns counts per bin"""
        pass

    # abstractmethod
    def pdf(self) -> ndarray:
        pass


class BinnedNormal(BinnedDistribution):
    """Samples correspond to gaussians restricted
    to the interval [1,...,n] and binned"""

    def __init__(self, n_bins: int, loc: ndarray, scale: ndarray) -> None:
        assert all([0 <= u < n_bins for u in loc])
        super().__init__(n_bins)
        self.loc = loc
        self.scale = scale
        self.batch_size = len(loc)

    def sample(self, size: int = 1) -> ndarray:
        z = self.loc + self.scale * stats.truncnorm.rvs(
            -self.loc / self.scale,
            (self.n_bins - 1.0 - self.loc) / self.scale,
            size=(size, self.batch_size),
        )
        out = np.zeros((self.batch_size, self.n_bins), int)
        for i in range(size):
            for j in range(self.batch_size):
                out[j, int(round(z[i, j]))] += 1

        return out

    def pdf(self) -> ndarray:
        x = np.arange(self.n_bins).astype(np.float)
        x = np.tile(x, (self.batch_size, 1))
        loc = np.expand_dims(self.loc, 1)
        scale = np.expand_dims(self.scale, 1)
        d = (1.0 / scale) * stats.truncnorm.pdf(
            (x - loc) / scale,
            -loc / scale,
            (self.n_bins - 1.0 - loc) / scale,
        )
        return d


class BinnedExponential(BinnedDistribution):
    """Samples correspond to exponentials restricted
    to the interval [1,...,n] and binned"""

    def __init__(self, n_bins: int, loc: ndarray, scale: ndarray) -> None:
        assert all([0 <= u < n_bins for u in loc])
        super().__init__(n_bins)
        self.loc = loc
        self.scale = scale
        self.batch_size = len(loc)

    def sample(self, size: int = 1) -> ndarray:
        z = self.loc + self.scale * stats.truncexpon.rvs(
            (self.n_bins - 1.0 - self.loc) / self.scale,
            size=(size, self.batch_size),
        )
        out = np.zeros((self.batch_size, self.n_bins), int)
        for i in range(size):
            for j in range(self.batch_size):
                out[j, int(round(z[i, j]))] += 1

        return out

    def pdf(self) -> ndarray:
        x = np.arange(self.n_bins).astype(np.float)
        x = np.tile(x, (self.batch_size, 1))
        loc = np.expand_dims(self.loc, 1)
        scale = np.expand_dims(self.scale, 1)
        d = (1.0 / scale) * stats.truncexpon.pdf(
            (x - loc) / scale, (self.n_bins - 1.0 - loc) / scale,
        )
        return d


class BinnedMixture(BinnedDistribution):
    """Samples a mixture distribution proportional to weights"""

    def __init__(
        self, distributions: List[BinnedDistribution], weights: ndarray,
    ) -> None:
        super().__init__(distributions[0].n_bins)
        self.batch_size = distributions[0].batch_size
        self.weights = weights
        self.distributions = distributions
        self.n_comps = len(weights)

    def sample(self, size: int = 1) -> ndarray:
        out = np.zeros((self.batch_size, self.n_bins), int)
        sample_sizes = np.random.multinomial(size, self.weights)

        for D, N in zip(self.distributions, sample_sizes):
            out += D.sample(max(N, 1))
        return out

    def pdf(self) -> ndarray:
        out = np.zeros((self.batch_size, self.n_bins))
        for D, w in zip(self.distributions, self.weights):
            out += w * D.pdf()
        return out


class WindowedDistribution(BinnedDistribution):
    """Fits a distribution in a window starting at 0 in
    a larger number of bins. Useful to have large regions with zeros"""

    def __init__(
        self, n_bins: int, distribution: BinnedDistribution, starts: list = [0]
    ):
        super().__init__(n_bins)
        self.distribution = distribution
        self.batch_size = distribution.batch_size
        if isinstance(starts, list) == 1:
            assert len(starts) == self.batch_size
            assert [s + distribution.n_bins <= n_bins for s in starts]
            self.starts = starts
        else:
            assert starts + distribution.n_bins <= n_bins
            self.starts = [starts] * self.batch_size
        self.window_size = distribution.n_bins

    def sample(self, size: int) -> ndarray:
        out = np.zeros((self.batch_size, self.n_bins), int)
        tmp = self.distribution.sample(size)
        for j, s in enumerate(self.starts):
            out[j, s : s + self.window_size] = tmp[j]
        return out

    def pdf(self) -> ndarray:
        out = np.zeros((self.batch_size, self.n_bins))
        tmp = self.distribution.pdf()
        for j, s in enumerate(self.starts):
            out[j, s : s + self.window_size] = tmp[j]
        return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    loc = np.array([0.0, 5.0, 7.0])
    scale = np.array([5.0, 10.0, 50.0])
    starts = [5, 10, 50]
    d = BinnedExponential(100, loc, scale)
    d = WindowedDistribution(200, d, starts)
    x = d.sample(size=10000)
    y = d.pdf()

    # plt.plot(y[0])
    # plt.plot(y[1])
    # plt.plot(y[2])
    # plt.show()
    # 0

    loc1 = np.array([0.0, 50.0])
    scale1 = np.array([5.0, 10.0])
    loc2 = np.array([25.0, 75.0])
    scale2 = np.array([7.0, 3.0])
    weights = np.array([0.25, 0.75])
    D1 = BinnedNormal(100, loc1, scale1)
    D2 = BinnedNormal(100, loc2, scale2)
    M = BinnedMixture([D1, D2], weights)

    x = M.sample(size=10000)
    y = M.pdf()
    plt.plot(y[0])
    plt.plot(y[1])
    plt.show()
    0
