import os
import numpy as np
from collections import deque
from functools import partial
import smoothdoq.binned as bd
import smoothdoq.noise as nm
import hydra
import math
from omegaconf import DictConfig
import ujson
from tqdm import tqdm


def make_distrib(cfg: DictConfig, n_bins: int) -> bd.BinnedDistribution:
    loc_rel = np.random.uniform(*cfg.loc)
    scale_rel = np.exp(np.random.uniform(*np.log(cfg.scale)))
    loc = np.array([loc_rel * (n_bins - 1)])
    scale = np.array([max(cfg.min_scale, scale_rel * (n_bins - 1))])

    if cfg.base_distrib == "exponential":
        base_distrib = bd.BinnedExponential(n_bins, loc, scale)
    elif cfg.base_distrib == "normal":
        base_distrib = bd.BinnedNormal(n_bins, loc, scale)
    else:
        raise NotImplementedError(cfg.base_distrib)

    return base_distrib


def add_noise(
    d: bd.BinnedDistribution, cfg: DictConfig
) -> bd.BinnedDistribution:

    if cfg.multiplicate_overdispersion:
        sigma = np.random.uniform(*cfg.od_sigma)
        d = nm.MultiplicativeDispersion(d, sigma)
    if cfg.rounding:
        stride = np.random.choice(cfg.rounding_stride)
        if stride > 1:
            d = nm.RoundingNoise(d, stride)
    if cfg.negbin_background:
        nr = np.random.uniform(*cfg.noise_ratio)
        dc = np.random.uniform(*cfg.disp_coef)
        d = nm.NegBinBackgroundNoise(d, nr, dc)

    noise_profile = dict(
        multiplicative_noise=cfg.multiplicate_overdispersion,
        od_sigma=sigma,
        rounding=cfg.rounding,
        rounding_stride=int(stride),
        negbin_background=cfg.negbin_background,
        noise_ratio=nr,
        disp_coef=dc,
    )

    return d, noise_profile


@hydra.main(config_name="sim_datasets_config")
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    buffer = deque(maxlen=cfg.batch_size)
    savedir = "data/simulated/" + cfg.config_name
    os.makedirs(savedir, exist_ok=True)

    # current options for these script are
    # Gaussian, Exponential, Mixture Gaussian,
    for i in tqdm(range(cfg.n_records)):
        M = cfg.n_bins_multiple_of
        n_bins = max(np.random.randint(*cfg.n_bins) // M, 1) * M
        window_size = np.random.uniform(*cfg.window_size)
        window_len = max(int(np.ceil(window_size * (n_bins - 1))), M // 2)
        window_start = np.random.randint(n_bins - window_len)

        n_comps = np.random.randint(*cfg.n_comps)
        alpha = np.random.uniform(*cfg.alpha)
        alpha_vec = np.full(n_comps, alpha)
        weights = np.random.dirichlet(alpha_vec)
        clean_distribs = [
            make_distrib(cfg, window_len) for j in range(n_comps)
        ]

        clean_distrib = bd.BinnedMixture(clean_distribs, weights)
        clean_pdf = bd.WindowedDistribution(
            n_bins, clean_distrib, [window_start]
        ).pdf()[0]
        clean_pdf_comps = [
            bd.WindowedDistribution(n_bins, d, [window_start]).pdf()[0]
            for d in clean_distribs
        ]

        distrib, profile = add_noise(clean_distrib, cfg)
        distrib = bd.WindowedDistribution(
            n_bins, distrib, starts=[window_start]
        )

        n_samples = np.random.randint(*cfg.n_obs)
        sample = distrib.sample(n_samples)[0]

        record = dict(
            n_samples=int(n_samples),
            sample=[int(x) for x in sample],
            base_distrib=cfg.base_distrib,
            weights=[int(w) for w in weights],
            pdf=clean_pdf.tolist(),
            pdf_comps=[d.tolist() for d in clean_pdf_comps],
            n_comps=int(n_comps),
            n_bins=int(n_bins),
            locs=[int(D.loc[0]) for D in clean_distribs],
            scales=[int(D.scale[0]) for D in clean_distribs],
            window_len=int(window_len),
            window_start=int(window_start),
            id=i,
            **profile,
        )
        buffer.append(record)

        if (i + 1) % cfg.batch_size == 0 or i == cfg.n_records - 1:
            fname = (
                f"{savedir}/{buffer[0]['id']:05d}-{buffer[-1]['id']:05d}.json"
            )
            with open(fname, "w") as io:
                ujson.dump(list(buffer), io)


if __name__ == "__main__":
    main()


#     def generate(self, n_bins=None):
#         if n_bins is None:
#             # L = np.log(self.n_bins[0])
#             # U = np.log(self.n_bins[1])
#             # n_bins = int(np.exp(self.rng.uniform(L, U)))
#             n_bins = int(self.rng.uniform(*self.n_bins))
#         n_obs = self.rng.randint(self.n_obs[0], self.n_obs[1] + 1)

#         #  sample the true signal component
#         # -- 2.a sample component weights
#         n_meanings = self.rng.randint(
#             self.n_meanings[0], self.n_meanings[1] + 1
#         )
#         # if n_bins < 50:
#         #     n_meanings = 1
#         if n_bins < 100:
#             n_meanings = min(2, n_meanings)
#         elif n_bins < 200:
#             n_meanings = min(3, n_meanings)
#         if self.rng.rand() < self.corner_prob:
#             offset = self.rng.choice((0, 2 * n_bins / (n_meanings + 1) - 1))
#         else:
#             offset = self.rng.uniform(0, 2 * n_bins / (n_meanings + 1) - 1)

#         # sample noise relative number meanings
#         max_noise = self.noise_ratio[1] / (n_meanings + 1)
#         L = self.noise_ratio[0]
#         U = max_noise
#         noise_ratio = self.rng.uniform(L, U)
#         mu_noise = np.ceil(noise_ratio * n_obs / n_bins)
#         noise_dispersion = self.rng.uniform(*self.noise_dispersion)
#         r = noise_dispersion
#         p = r / (r + mu_noise)
#         noise = self.rng.negative_binomial(r, p, n_bins)
#         #

#         alpha_meanings = self.rng.uniform(*self.alpha_comps)
#         while True:
#             dirichlet_params = np.full(n_meanings, alpha_meanings)
#             wts_meanings = self.rng.dirichlet(dirichlet_params)
#             if wts_meanings.min() > 0.1:
#                 break
#         meaning_n_comps = []
#         meaning_pdfs = []
#         meaning_signals = []
#         meaning_comp_counts = []
#         meaning_n_obs = []
#         meaning_rounding = []
#         meaning_means = []
#         meaning_centroids = []
#         meaning_sigmas = []

#         for i, mw in enumerate(wts_meanings):
#             #
#             if not self.exact:
#                 rounding = min(
#                     self.rng.choice(self.rounding), max(n_bins // 5, 1)
#                 )
#                 n_comps = self.rng.randint(
#                     self.n_comps[0], self.n_comps[1] + 1
#                 )
#             else:
#                 rounding = 1
#                 n_comps = 1
#             meaning_rounding.append(rounding)
#             # obtain number of components and weights
#             alpha = self.rng.uniform(*self.alpha_comps)
#             wts = self.rng.dirichlet((alpha,) * n_comps)
#             # -- 2.b sample normal for each component
#             n_signal = int(np.round(np.ceil((1.0 - noise_ratio) * n_obs * mw)))
#             comp_obs = [int(np.ceil(w * n_signal)) for w in wts]
#             # define valid mean range for meaning
#             delta = n_bins / (n_meanings + 1)
#             m0 = offset + i * delta
#             L, U = m0 - 0.075 * delta, m0 + 0.075 * delta
#             # sample means
#             mean0 = self.rng.uniform(max(L, 0.0), min(U, n_bins - 1.0))
#             if not self.exact:
#                 sbr = self.max_sigma_to_bins_ratio
#                 L = self.log_sigmas[0]
#                 U = min(self.log_sigmas[1], np.log(sbr * n_bins))
#                 if L >= U:
#                     print("Warning: error in bounds for sigma")
#                 sigma0 = np.exp(self.rng.uniform(L, U))
#                 sigma0 = max(sigma0, 0.5 * rounding)
#                 if n_comps == 1:
#                     means = np.array([mean0])
#                     sigmas = np.array([sigma0])
#                 else:
#                     # all other comps are shape factors so mus be close enough
#                     means1 = []
#                     sigmas1 = []
#                     curr_min_logsigma = np.log(sigma0)
#                     cur_max_logsigma = np.log(sigma0)
#                     for _ in range(n_comps - 1):
#                         # any subcomponent must be max one std of distance
#                         attempts = 0
#                         inner_mean_dist = 0.5 * delta
#                         while attempts < self.max_attempts:
#                             m = self.rng.normal(mean0, inner_mean_dist)
#                             isr = self.inner_sigma_ratio
#                             L = max(
#                                 (1.0 / isr) * curr_min_logsigma,
#                                 self.log_sigmas[0],
#                             )
#                             U = min(
#                                 isr * cur_max_logsigma,
#                                 self.log_sigmas[1],
#                                 np.log(0.5 * sbr * delta),
#                             )
#                             U = max(L, U)
#                             if L > U:
#                                 print(
#                                     "Warning: error in bounds for sigma (in)"
#                                 )
#                             s = np.exp(self.rng.uniform(L, U))
#                             eta = self.inner_mode_dist_tol
#                             if np.abs(m - mean0) < eta * max(s, sigma0):
#                                 means1.append(m)
#                                 sigmas1.append(s)
#                                 curr_min_logsigma = min(s, curr_min_logsigma)
#                                 cur_max_logsigma = max(s, cur_max_logsigma)
#                                 break
#                             else:
#                                 inner_mean_dist *= 0.8
#                             attempts += 1
#                         if attempts == 25:
#                             raise RuntimeError(
#                                 "Warning: max attempts for mean, sigma (in)"
#                             )
#                     means = np.array([mean0] + means1)
#                     sigmas = np.array([sigma0] + sigmas1)
#             else:
#                 means = np.array([mean0])
#                 sigmas = np.array([0.0])
#             centroid = np.dot(wts, means)
#             #
#             meaning_means.append(means)
#             meaning_sigmas.append(sigmas)
#             # now obtain mean counts, everythign as in the case of negbin.py
#             bincounts = []
#             pdfs = []
#             comp_counts = []
#             xbins = np.arange(n_bins, dtype=float) + 0.5
#             for mi, si, ni in zip(means, sigmas, comp_obs):
#                 S = si
#                 N = ni
#                 if not self.exact:
#                     dens = norm.pdf(xbins, loc=mi, scale=si)
#                     while True:
#                         sample = self.rng.normal(mi, S, size=N)
#                         if rounding > 1:
#                             r = rounding
#                             sample = mi + np.round((sample - mi) / r) * r
#                         counts = np.zeros(n_bins, dtype=int)
#                         for j in np.round(sample).astype(int):
#                             if 0 <= j < n_bins:
#                                 counts[j] += 1
#                         M = counts.sum()
#                         if M > 0:
#                             comp_counts.append(M)
#                             break
#                         else:
#                             S *= 2.0
#                             N *= 2
#                 else:
#                     dens = np.zeros(n_bins)
#                     mi_loc = round(int(np.clip(mi, 0, n_bins - 1)))
#                     dens[mi_loc] = 1.0
#                     comp_counts.append(N)
#                     counts = np.zeros(n_bins, dtype=int)
#                     counts[mi_loc] = N
#                 # evaluate and trim density, necessary after rounding
#                 if self.trim_ends_with_no_data and not self.exact:
#                     j = 0
#                     while j < n_bins and counts[j] == 0:
#                         dens[j] = 1e-10
#                         j += 1
#                     j = n_bins - 1
#                     while j >= 0 and counts[j] == 0:
#                         dens[j] = 1e-10
#                         j -= 1
#                     # normalize density as probability
#                     if dens.sum() < 1e-6:  # std too small!
#                         dens[int(np.round(np.clip(mi, 0, n_bins - 1)))] = 1.0

#                 # add to data
#                 bincounts.append(counts)
#                 pdfs.append(dens)

#             # totals
#             comp_counts = np.array(comp_counts, dtype=int)
#             wts_ = np.array(comp_counts, dtype=float) / sum(comp_counts)
#             pdf = sum(w * s for w, s in zip(wts_, pdfs))
#             if pdf.sum() == 0:
#                 print("Warning: pdf sum is zero???")
#             pdf /= pdf.sum()
#             total_obs = comp_counts.sum()
#             signal = sum(s for s in bincounts)

#             # add two meaning data
#             meaning_signals.append(signal)
#             meaning_centroids.append(centroid)
#             meaning_pdfs.append(pdf)
#             meaning_n_comps.append(n_comps)
#             meaning_comp_counts.append(comp_counts)
#             meaning_n_obs.append(total_obs)
#         meaning_n_obs = np.array(meaning_n_obs)
#         meaning_n_comps = np.array(meaning_n_comps)

#         # aggregate all meanings
#         signal = sum(s for s in meaning_signals)
#         disp_coef = self.rng.uniform(*self.signal_dispersion)
#         disp = np.exp(self.rng.normal(0.0, disp_coef, n_bins))
#         signal_noisy = np.round(disp * signal)
#         obs = signal_noisy + noise

#         # modes and post-trimming weights
#         modes_onehot = np.zeros(n_bins, dtype=int)
#         centroids_onehot = np.zeros(n_bins, dtype=int)
#         # meaning_wts = meaning_n_obs / meaning_n_obs.sum()
#         meaning_wts = wts_meanings
#         meaning_wts_onehot = np.zeros(n_bins)
#         meaning_modes = []
#         for w, d, c in zip(meaning_wts, meaning_pdfs, meaning_centroids):
#             mode = np.argmax(d)
#             meaning_modes.append(mode)
#             modes_onehot[mode] = 1
#             meaning_wts_onehot[mode] = w
#             centroids_onehot[int(np.clip(np.round(c), 0, n_bins - 1))] = 1
#         meaning_modes = np.array(meaning_modes)

#         # total signal density
#         pdf = sum(w * s for w, s in zip(meaning_pdfs, meaning_wts))
#         pdf /= pdf.sum()
#         total_noise = noise.sum()
#         total_obs = sum(meaning_n_obs)
#         expected = pdf * total_obs

#         # 3. return output
#         output = {
#             "n_bins": n_bins,
#             "noise": noise,
#             "signal": signal,
#             "signal_noisy": signal_noisy,
#             "obs": obs,
#             "meaning_modes": meaning_modes,
#             "meaning_centroids": meaning_centroids,
#             "meaning_n_obs": meaning_n_obs,
#             "meaning_n_comps": meaning_n_comps,
#             "modes_onehot": modes_onehot,
#             "meaning_wts": meaning_wts,
#             "meaning_wts_onehot": meaning_wts_onehot,
#             "meaning_means": meaning_means,
#             "centroids": meaning_centroids,
#             "centroids_onehot": centroids_onehot,
#             "meaning_sigmas": meaning_sigmas,
#             "n_obs": total_obs,
#             "n_meanings": n_meanings,
#             "n_bins": n_bins,
#             "pdf": pdf,
#             "meaning_pdfs": meaning_pdfs,
#             "noise_dispersion": noise_dispersion,
#             "signal_overdispersion": disp_coef,
#             "noise_ratio": noise_ratio,
#             "total_noise": total_noise,
#             "total_obs": total_obs,
#             "expected": expected,
#             "rounding": meaning_rounding,
#         }
#         return output


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from smoothdoq.binned_distributions import BinnedExponential

#     loc = np.array([0.0, 5.0, 7.0])
#     scale = np.array([5.0, 10.0, 50.0])
#     D = BinnedExponential(100, loc, scale)
#     D = RoundingNoise(D, 5)
#     x = D.sample(size=10000)

#     # plt.bar(range(100), x[2])
#     # plt.show()

#     D = BinnedExponential(100, loc, scale)
#     D = MultiplicativeDispersion(D, 0.8)
#     x = D.sample(size=10000, adjust_size=True)

#     # plt.bar(range(100), x[2])
#     # plt.show()

#     # D = BinnedExponential(100, loc, scale / 5.0)
#     D = NegBinBackgroundNoise(D, noise_ratio=0.25, disp_coef=0.25)
#     x = D.sample(size=10000, adjust_size=True)
#     y = D.pdf()

#     plt.bar(range(100), x[1] / x[1].sum(), label="samples")
#     plt.plot(range(100), y[1] / y[1].sum(), label="density")
#     plt.legend()
#     plt.show()

#     0
