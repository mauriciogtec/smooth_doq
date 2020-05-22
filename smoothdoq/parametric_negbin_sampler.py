import numpy as np
from scipy.stats import norm
# import matplotlib.pyplot as plt


class RecordGenerator():
    def __init__(self,
                 seed=None,
                 n_bins=(10, 500),
                 n_obs=(1000, 100000),
                 noise_ratio=(0.001, 0.75),
                 noise_dispersion=(0.05, 2.0),
                 n_comps=(1, 5),
                 n_meanings=(1, 4),
                 alpha_meanings=(0.5, 4.0),
                 alpha_comps=(0.5, 4.0),
                 rounding=[1, 2, 5, 10],
                 max_sigma_to_bins_ratio=1.0,
                 inner_sigma_ratio=3.0,
                 sigmas=(0.1, 25.0),
                 inner_mode_dist_tol=0.25,
                 signal_dispersion=(0.00001, 0.4),
                 trim_ends_with_no_data=True,
                 max_attempts=100,
                 corner_prob=0.0,
                 exact=False):
        self.max_attempts = max_attempts
        self.rng = np.random.RandomState(seed)
        self.n_bins = n_bins
        self.n_obs = n_obs
        self.noise_ratio = noise_ratio
        self.noise_dispersion = noise_dispersion
        self.n_points = n_obs
        self.n_meanings = n_meanings
        self.n_comps = n_comps
        self.inner_mode_dist_tol = inner_mode_dist_tol
        self.alpha_meanings = alpha_meanings
        self.alpha_comps = alpha_comps
        self.sigmas = sigmas
        self.log_sigmas = (np.log(sigmas[0]), np.log(sigmas[1]))
        self.signal_dispersion = signal_dispersion
        self.rounding = rounding
        self.max_sigma_to_bins_ratio = max_sigma_to_bins_ratio
        self.inner_sigma_ratio = inner_sigma_ratio
        self.trim_ends_with_no_data = trim_ends_with_no_data
        self.corner_prob = corner_prob
        self.exact = exact

    def generate(self, n_bins=None):
        if n_bins is None:
            # L = np.log(self.n_bins[0])
            # U = np.log(self.n_bins[1])
            # n_bins = int(np.exp(self.rng.uniform(L, U)))
            n_bins = int(self.rng.uniform(*self.n_bins))
        n_obs = self.rng.randint(self.n_obs[0], self.n_obs[1] + 1)

        #  sample the true signal component
        # -- 2.a sample component weights
        n_meanings = self.rng.randint(self.n_meanings[0], self.n_meanings[1]+1)
        # if n_bins < 50:
        #     n_meanings = 1
        if n_bins < 100:
            n_meanings = min(2, n_meanings)
        elif n_bins < 200:
            n_meanings = min(3, n_meanings)
        if self.rng.rand() < self.corner_prob:
            offset = self.rng.choice((0, 2 * n_bins / (n_meanings + 1) - 1))
        else:
            offset = self.rng.uniform(0, 2 * n_bins / (n_meanings + 1) - 1)

        # sample noise relative number meanings
        max_noise = self.noise_ratio[1] / (n_meanings + 1)
        L = self.noise_ratio[0]
        U = max_noise
        noise_ratio = self.rng.uniform(L, U)
        mu_noise = np.ceil(noise_ratio * n_obs / n_bins)
        noise_dispersion = self.rng.uniform(*self.noise_dispersion)
        r = noise_dispersion
        p = r / (r + mu_noise)
        noise = self.rng.negative_binomial(r, p, n_bins)
        #

        alpha_meanings = self.rng.uniform(*self.alpha_comps)
        while True:
            dirichlet_params = np.full(n_meanings, alpha_meanings)
            wts_meanings = self.rng.dirichlet(dirichlet_params)
            if wts_meanings.min() > 0.1:
                break
        meaning_n_comps = []
        meaning_pdfs = []
        meaning_signals = []
        meaning_comp_counts = []
        meaning_n_obs = []
        meaning_rounding = []
        meaning_means = []
        meaning_centroids = []
        meaning_sigmas = []
        
        for i, mw in enumerate(wts_meanings):
            #
            if not self.exact:
                rounding = min(self.rng.choice(self.rounding), max(n_bins // 5, 1))
                n_comps = self.rng.randint(self.n_comps[0], self.n_comps[1] + 1)
            else:
                rounding = 1
                n_comps = 1
            meaning_rounding.append(rounding)
            # obtain number of components and weights
            alpha = self.rng.uniform(*self.alpha_comps)
            wts = self.rng.dirichlet((alpha, ) * n_comps)
            # -- 2.b sample normal for each component
            n_signal = int(np.round(np.ceil((1.0 - noise_ratio) * n_obs * mw)))
            comp_obs = [int(np.ceil(w * n_signal)) for w in wts]
            # define valid mean range for meaning
            delta = n_bins / (n_meanings + 1)
            m0 = offset + i * delta
            L, U = m0 - 0.075 * delta, m0 + 0.075 * delta
            # sample means
            mean0 = self.rng.uniform(max(L, 0.0), min(U, n_bins - 1.0))
            if not self.exact:
                sbr = self.max_sigma_to_bins_ratio
                L = self.log_sigmas[0]
                U = min(
                    self.log_sigmas[1],
                    np.log(sbr * n_bins))
                if L >= U:
                    print("Warning: error in bounds for sigma")
                sigma0 = np.exp(self.rng.uniform(L, U))
                sigma0 = max(sigma0, 0.5 * rounding)
                if n_comps == 1:
                    means = np.array([mean0])
                    sigmas = np.array([sigma0])
                else:
                    # all other comps are shape factors so mus be close enough
                    means1 = []
                    sigmas1 = []
                    curr_min_logsigma = np.log(sigma0)
                    cur_max_logsigma = np.log(sigma0)
                    for _ in range(n_comps - 1):
                        # any subcomponent must be max one std of distance
                        attempts = 0
                        inner_mean_dist = 0.5 * delta
                        while attempts < self.max_attempts:
                            m = self.rng.normal(mean0, inner_mean_dist)
                            isr = self.inner_sigma_ratio
                            L = max(
                                (1.0 / isr) * curr_min_logsigma,
                                self.log_sigmas[0])
                            U = min(
                                isr * cur_max_logsigma,
                                self.log_sigmas[1],
                                np.log(0.5 * sbr * delta))
                            U = max(L, U)
                            if L > U:
                                print("Warning: error in bounds for sigma (in)")
                            s = np.exp(self.rng.uniform(L, U))
                            eta = self.inner_mode_dist_tol
                            if np.abs(m - mean0) < eta * max(s, sigma0):
                                means1.append(m)
                                sigmas1.append(s)
                                curr_min_logsigma = min(s, curr_min_logsigma)
                                cur_max_logsigma = max(s, cur_max_logsigma)
                                break
                            else:
                                inner_mean_dist *= 0.8
                            attempts += 1
                        if attempts == 25:
                            raise RuntimeError("Warning: max attempts for mean, sigma (in)")
                    means = np.array([mean0] + means1)
                    sigmas = np.array([sigma0] + sigmas1)
            else:
                means = np.array([mean0])
                sigmas = np.array([0.0])
            centroid = np.dot(wts, means)
            #
            meaning_means.append(means)
            meaning_sigmas.append(sigmas)
            # now obtain mean counts, everythign as in the case of negbin.py
            bincounts = []
            pdfs = []
            comp_counts = []
            xbins = np.arange(n_bins, dtype=float) + 0.5
            for mi, si, ni in zip(means, sigmas, comp_obs):
                S = si
                N = ni
                if not self.exact:
                    dens = norm.pdf(xbins, loc=mi, scale=si)
                    while True:
                        sample = self.rng.normal(mi, S, size=N)
                        if rounding > 1:
                            r = rounding
                            sample = mi + np.round((sample - mi) / r) * r
                        counts = np.zeros(n_bins, dtype=int)
                        for j in np.round(sample).astype(int):
                            if 0 <= j < n_bins:
                                counts[j] += 1
                        M = counts.sum()
                        if M > 0:
                            comp_counts.append(M)
                            break
                        else:
                            S *= 2.0
                            N *= 2
                else:
                    dens = np.zeros(n_bins)
                    mi_loc = round(int(np.clip(mi, 0, n_bins - 1)))
                    dens[mi_loc] = 1.0
                    comp_counts.append(N)
                    counts = np.zeros(n_bins, dtype=int)
                    counts[mi_loc] = N
                # evaluate and trim density, necessary after rounding
                if self.trim_ends_with_no_data and not self.exact:
                    j = 0
                    while j < n_bins and counts[j] == 0:
                        dens[j] = 1e-10
                        j += 1
                    j = n_bins - 1
                    while j >= 0 and counts[j] == 0:
                        dens[j] = 1e-10
                        j -= 1
                    # normalize density as probability
                    if dens.sum() < 1e-6:   # std too small!
                        dens[int(np.round(np.clip(mi, 0, n_bins - 1)))] = 1.0

                # add to data
                bincounts.append(counts)
                pdfs.append(dens)

            # totals
            comp_counts = np.array(comp_counts, dtype=int)
            wts_ = np.array(comp_counts, dtype=float) / sum(comp_counts)
            pdf = sum(w * s for w, s in zip(wts_, pdfs))
            if pdf.sum() == 0:
                print("Warning: pdf sum is zero???")
            pdf /= pdf.sum()
            total_obs = comp_counts.sum()
            signal = sum(s for s in bincounts)

            # add two meaning data
            meaning_signals.append(signal)
            meaning_centroids.append(centroid)
            meaning_pdfs.append(pdf)
            meaning_n_comps.append(n_comps)
            meaning_comp_counts.append(comp_counts)
            meaning_n_obs.append(total_obs)
        meaning_n_obs = np.array(meaning_n_obs)
        meaning_n_comps = np.array(meaning_n_comps)

        # aggregate all meanings
        signal = sum(s for s in meaning_signals)
        disp_coef = self.rng.uniform(*self.signal_dispersion)
        disp = np.exp(self.rng.normal(0.0, disp_coef, n_bins))
        signal_noisy = np.round(disp * signal)
        obs = signal_noisy + noise

        # modes and post-trimming weights
        modes_onehot = np.zeros(n_bins, dtype=int)
        centroids_onehot = np.zeros(n_bins, dtype=int)
        # meaning_wts = meaning_n_obs / meaning_n_obs.sum()
        meaning_wts = wts_meanings
        meaning_wts_onehot = np.zeros(n_bins)
        meaning_modes = []
        for w, d, c in zip(meaning_wts, meaning_pdfs, meaning_centroids):
            mode = np.argmax(d)
            meaning_modes.append(mode)
            modes_onehot[mode] = 1
            meaning_wts_onehot[mode] = w
            centroids_onehot[int(np.clip(np.round(c), 0, n_bins - 1))] = 1
        meaning_modes = np.array(meaning_modes)

        # total signal density
        pdf = sum(w * s for w, s in zip(meaning_pdfs, meaning_wts))
        pdf /= pdf.sum()
        total_noise = noise.sum()
        total_obs = sum(meaning_n_obs)
        expected = pdf * total_obs

        # 3. return output
        output = {
            'n_bins': n_bins,
            'noise': noise,
            'signal': signal,
            'signal_noisy': signal_noisy,
            'obs': obs,
            'meaning_modes': meaning_modes,
            'meaning_centroids': meaning_centroids,
            'meaning_n_obs': meaning_n_obs,
            'meaning_n_comps': meaning_n_comps,
            'modes_onehot': modes_onehot,
            'meaning_wts': meaning_wts,
            'meaning_wts_onehot': meaning_wts_onehot,
            'meaning_means': meaning_means,
            'centroids': meaning_centroids,
            'centroids_onehot': centroids_onehot,
            'meaning_sigmas': meaning_sigmas,
            'n_obs': total_obs,
            'n_meanings': n_meanings,
            'n_bins': n_bins,
            'pdf': pdf,
            'meaning_pdfs': meaning_pdfs,
            'noise_dispersion': noise_dispersion,
            'signal_overdispersion': disp_coef,
            'noise_ratio': noise_ratio,
            'total_noise': total_noise,
            'total_obs': total_obs,
            'expected': expected,
            'rounding': meaning_rounding
        }
        return output

# # %%


# simulator = RecordGenerator()


# # %%

# data = simulator.generate()

# x = np.arange(data['n_bins'])
# plt.bar(x, data['counts'], width=1.0)
# plt.plot(x, data['expected'], c="red")


# # %%
# def normalize(x):
#     x = x / sum(x)
#     return x


# resids = normalize(data['signal_normalized']) - normalize(data['counts'])
# plt.scatter(x, resids)
# plt.axhline(0, c="red")
# plt.title("residuals")

# # %%
