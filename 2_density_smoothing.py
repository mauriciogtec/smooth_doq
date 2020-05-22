# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
# wdir = "./"
wdir = "./"
import sys
sys.path.append(wdir)

from smoothdoq.gan import Generator, Discriminator, Features, SignalHead, DeconvHead2
from smoothdoq.parametric_negbin_sampler import RecordGenerator

from scipy import signal
from scipy.special import expit, softmax

import pdb
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd
import glob
from collections import defaultdict
import seaborn as sns
import pdb

sns.set_style("whitegrid")

from matplotlib import cm
viridis = cm.get_cmap('viridis', 12)
# cols = ["#926cb6", "#93b793", "#d31d00", "#ff900d", "#fefb03", "black"]
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# %% [markdown]
# ## Load the data

# %%
# fdir = os.path.join(wdir, "./data/batches/")
# files = glob.glob(fdir + "batch*.json")

# dat = defaultdict(dict)
# for file in tqdm(files):
#     with open(file, "r") as f:
#         for D in json.load(f):
#             obj, dim, l, u, c = D
#             Dprime = {'obj': obj, 'dim': dim, "loglower": l,
#                       'logupper': u, "counts": c}
#             dat[obj][dim] = Dprime


# %%
# example how to get attributes from object
# dat['person'].keys()

# %% [markdown]
# ## Examples of smoothing and context finding
# %% [markdown]
# ### 1. Load Neural Network Models

# %%
inputs_smoother = tf.keras.Input(shape=(None, 1), dtype=tf.float32)
inputs_deconv = tf.keras.Input(shape=(None, 2), dtype=tf.float32)

feats_smoother = Features(ksize=7, filters=32, nblocks=12, batch_norms=False)(inputs_smoother)
feats_deconv = Features(ksize=7, filters=32, nblocks=12, batch_norms=False)(inputs_deconv)

signal_head = SignalHead(ksize=5, filters=32, nblocks_signal=2, nblocks_peaks=4, batch_norms=False)
deconv_head =  DeconvHead2(ksize=5, filters=32, nblocks_signal=2, nblocks_weights=4, batch_norms=False)

outputs_signals, outputs_peaks = signal_head([feats_smoother, inputs_smoother])
outputs_deconv, outputs_wts = deconv_head([feats_deconv, inputs_deconv])

smoother = tf.keras.Model(inputs=inputs_smoother, outputs=[outputs_signals, outputs_peaks])
deconv = tf.keras.Model(inputs=inputs_deconv, outputs=[outputs_deconv, outputs_wts])


# %%
smoother.load_weights(os.path.join(wdir, "trained/smoother.h5"))
deconv.load_weights(os.path.join(wdir, "trained/deconv.h5"))

# %% [markdown]
# ### 2. Denoise, Deconvolute and Visualize

# %%
def preprocess(cnts):
    x = cnts
    if len(x) > 10 and x.sum() > 2 * len(x):
        x = np.maximum(0.0, cnts - 2.0) + 1e-6
    x /= x.sum()
    x *= len(x)
    x = np.reshape(x, (1, -1, 1))
    return x

def postprocess(dens, threshold=0.1):
    nbins = dens.shape[-1]
    x = dens
    x = np.maximum(x - threshold / nbins, 0.0)
    if np.any(x.sum(-1) == 0):
        return postprocess(dens, threshold=0.5*threshold)
    x /= x.sum(-1, keepdims=True)
    return x

# %%
def plotdoq(obj, dim, savepath=None, rounding=1):
    dim = dim.upper()
    obj = obj.replace("-", " ")
    d = dat[obj][dim]
    u = d['logupper']
    x = np.array(d['counts'], np.float32)
    nbins = len(x)
    #
    h = np.arange(len(x))
    plt.bar(h, x, width=1.3, alpha=0.5, linewidth=0.0, color="blue")
    locs = [int((i) * (nbins - 1)) // 4 for i in range(5)]
    rfun = lambda x: round(10**x) if rounding == 1 else np.round(10**x, rounding)
    labs = [f"{rfun(u[j]):,}" for j in locs]
    plt.xticks(locs, labs, rotation=0, size=15)
    plt.yticks(size=15)
#     plt.ylabel("counts", fontsize=16)
#     plt.xlabel(dim.lower(), fontsize=16)
    if savepath is not None:
        plt.savefig(savepath, pad_inches=0.1, bbox_inches="tight")
    plt.show()
    
    
    
def plotdoq_smooth(obj, dim,
                   savepath=None,
                   rounding=1, 
                   peaks_prominence=0.001,
                   peaks_min_rel_height=0.1,
                   peaks_min_rel_distance=0.05,
                   peaks_rel_threshold=0.001,
                   tick_rotation=15):
    dim = dim.upper()
    obj = obj.replace("-", " ")
    d = dat[obj][dim]
    u = d['logupper']
    x = np.array(d['counts'], np.float32)
    nbins = len(x)
    smoother_input = preprocess(x)
    yhat, mhat = smoother(smoother_input)
    yhat = postprocess(yhat.numpy().squeeze())
    peaks = signal.find_peaks(
        yhat,
        height=peaks_min_rel_height / nbins,
        distance=peaks_min_rel_distance * nbins, 
        threshold=peaks_rel_threshold / nbins,
        prominence=peaks_prominence)
    #
    peaks = peaks[0]
    #
    h = np.arange(len(x))
    plt.bar(h, x / x.sum(), width=1.0, alpha=0.5, linewidth=0.0, color="blue")
    plt.plot(h, yhat, c="black", lw=4) 
    plt.plot(peaks, yhat[peaks], "x", color="red", lw=3.0, markersize=15) 
    locs = [int((i) * (nbins - 1)) // 4 for i in range(5)]
    labs = [f"{int(np.round(10 ** u[j])):,}" for j in locs]
    plt.xticks(locs, labs, rotation=tick_rotation, size=15)
    plt.yticks(size=15)
    if savepath is not None:
        plt.savefig(savepath, pad_inches=0.1, bbox_inches="tight")
    plt.show()
    
    
def plotdoq_deconv(obj, dim, 
                   savepath=None,
                   rounding=1, 
                   peaks_prominence=0.001,
                   peaks_min_rel_height=0.1,
                   peaks_min_rel_distance=0.05,
                   peaks_rel_threshold=0.001,
                   tick_rotation=15):
    dim = dim.upper()
    obj = obj.replace("-", " ")
    d = dat[obj][dim]
    u = d['logupper']
    x = np.array(d['counts'], np.float32)
    nbins = len(x)
    smoother_input = preprocess(x)
    yhat, mhat = smoother(smoother_input)
    yhat = postprocess(yhat.numpy().squeeze())
    mhat = expit(np.squeeze(mhat.numpy()))
    #
    peaks = signal.find_peaks(
        yhat,
        height=peaks_min_rel_height / nbins,
        distance=peaks_min_rel_distance * nbins, 
        threshold=peaks_rel_threshold / nbins,
        prominence=peaks_prominence)
    #
    peaks = peaks[0]
    deconv_input = np.zeros((len(peaks), nbins,1), np.float32)
    for j, p in enumerate(peaks):
        deconv_input[j, p, 0] = 1.0
    counts_input = np.concatenate(len(peaks) * [smoother_input], 0)
    deconv_input = np.concatenate([counts_input, deconv_input], -1)
    deconv_signals, wts = deconv(deconv_input)
    deconv_signals = deconv_signals.numpy().squeeze(-1)
    wts = wts.numpy().squeeze(-1)
    wts = softmax(wts)
    #
    h = np.arange(len(x))
    plt.bar(h, x / x.sum(), width=1.0, alpha=0.35, linewidth=0.0, color="blue")
    legs = []
    for j, p in enumerate(peaks):
        dhat = postprocess(deconv_signals[j, :])
        what = wts[j]
        l = f"{j + 1} ({np.round(100.0 * what,1)}%)"
        legs.append(l)
        plt.plot(h, dhat, lw=4, label=l) 
    locs = list(peaks)
    rfun = lambda x: round(10**x) if rounding == 1 else np.round(10**x, rounding)
    labs = [f"{rfun(u[j]):,}" for j in locs]
    plt.xticks(locs, labs, rotation=tick_rotation, size=15)
    plt.yticks(size=15)
    plt.legend(title="context weight", fontsize=15, title_fontsize=15)
    if savepath is not None:
        plt.savefig(savepath, pad_inches=0.1, bbox_inches="tight")
    plt.show()


# %%
# obj = "full time"
# dim = "currency"
# plotdoq(obj, dim, savepath=f"./examples/{obj}_{dim}_counts.pdf")
# plotdoq_smooth(obj, dim, savepath=f"./examples/{obj}_{dim}_smooth.pdf")
# plotdoq_deconv(obj, dim, rounding=2, savepath=f"./examples/{obj}_{dim}_contexts.pdf")


# %%
obj_dim_list = [
    ("breakfast", "time"),
    ("camera", "currency"),
    ("marathon", "duration"),
    ("marathon", "length"),
#     ("half-marathon", "length"),
#     ("roar", "duration"),
#     ("cancer", "duration"),
#     ("peanut", "mass"),
    ("peanut", "length"),
#     ("person", "mass"),
#     ("person", "length"),
#     ("baby", "mass"),
#     ("baby", "length"),
    ("sun", "temperature"),
#     ("neurons", "frequency"),
#     ("groceries", "duration"),
#     ("software-engineers", "currency"),
#     ("swordfish", "mass"),
#     ("football", "mass"),
#     ("stockade", "length"),
#     ("full-time", "currency"),
#     ("doors", "length"),
#     ("door", "length"),
#     ("door", "mass"),
#     ("surgery", "duration"),
#     ("idea", "currency"),
#     ("1Mhz", "frequency"),
#     ("solutions", "currency"),
#     ("snow", "temperature"),
#     ("elephant", "mass"),
#     ("cow", "mass"),
#     ("lion", "mass"),
#     ("elephant", "length"),
#     ("cow", "length"),
#     ("lion", "length"),
#     ("car", "speed"),
#     ("person", "speed"),
#     ("cheetah", "speed"),
#     ("sun", "temperature"),
#     ("sun", "mass"),
]

# for obj, dim in obj_dim_list:
#     print(f"\n--- {obj} {dim} ---")
#     if obj == "peanut":
#         plotdoq(obj, dim, 
#                 savepath=f"./examples/{obj}_{dim}_counts.pdf")
#     plotdoq_smooth(obj, dim, 
#                    savepath=f"./examples/{obj}_{dim}_smooth.pdf", 
#                    peaks_prominence=0.002, 
#                    peaks_rel_threshold=0.0,
#                    tick_rotation=30)
#     plotdoq_deconv(obj, dim, rounding=2, 
#                    savepath=f"./examples/{obj}_{dim}_contexts.pdf", 
#                    peaks_prominence=0.002, 
#                    peaks_rel_threshold=0.0,
#                    tick_rotation=30)

# %% [markdown]
# ## 2. GAN Examples

# %%
# geninput_noise = tf.keras.Input(shape=(None, 1), dtype=tf.float32)
# geninput_signal = tf.keras.Input(shape=(None, 1), dtype=tf.float32)
# geninput_noiseratio = tf.keras.Input(shape=(1, 1), dtype=tf.float32)
# disinput = tf.keras.Input(shape=(None, 1), dtype=tf.float32)

# genoutput = Generator(ksize=5, filters=48, nblocks=16, batch_norms=False)([geninput_noise, geninput_signal, geninput_noiseratio])
# disoutput = Discriminator(ksize=7, filters=32, nblocks=16, batch_norms=False)(disinput)

# generator = tf.keras.Model(inputs=[geninput_noise, geninput_signal, geninput_noiseratio], outputs=genoutput)
# discriminator = tf.keras.Model(inputs=disinput, outputs=disoutput)


# # %%
# generator.load_weights(os.path.join(wdir, "ckpts/generator_0125.h5"))
# discriminator.load_weights(os.path.join(wdir, "ckpts/discriminator_0125.h5"))


# # %%
# def plotfake(fake_mode,
#              sim_mode=1,
#              rounding=1,
#              savepath=None,
#              n_bins=None,
#              show_pdf=False,
#              seed=None,
#              nr=0.25):
#     rng = np.random.RandomState(10 * seed)
#     assert fake_mode in ("gan", "param")

#     if sim_mode == 1:
#         # this one draws from true signals via mixture models
#         simulator = RecordGenerator(
#             n_obs=(int(10000 / nr), int(10000 / nr) + 1),
#             n_meanings=(1, 4),
#             n_comps=(1, 10),
#             noise_ratio=(nr, nr + 1e-6),
#             noise_dispersion=(0.05, 1.5),
#             alpha_meanings=(1.0, 4.0),
#             rounding=[rounding],
#             inner_mode_dist_tol=0.15,
#             inner_sigma_ratio=3.0,
#             max_sigma_to_bins_ratio=0.35,
#             sigmas=(0.1, 100.0),
#             corner_prob=0.01,
#             seed=seed)
#     else:
#         simulator = RecordGenerator(
#             exact=True,
#             n_obs=(int(10000 / nr), int(10000 / nr) + 1),
#             n_meanings=(1, 1),
#             noise_ratio=(nr, nr + 1e-6),
#             signal_dispersion=(0.0, 1e-6),
#             noise_dispersion=(0.05, 1.5),
#             seed=seed)
    
#     sim = simulator.generate(n_bins=n_bins)
#     if n_bins is None:
#         n_bins = sim['n_bins']
#     d = np.array(sim['pdf'], np.float32)
#     d_normalized = np.reshape(d * n_bins, (1, -1, 1))
#     if fake_mode == "gan":
#         z = np.reshape(rng.normal(size=n_bins), (1, -1, 1))
#         z = z.astype(np.float32)
#         nr = np.array([[[nr]]], dtype=np.float32)
#         fake = generator([z, d_normalized, nr])
#         fake = fake.numpy().squeeze()
#     else:
#         fake = sim['obs']
#         fake = np.array(fake)
#         fake /= fake.sum()
#     fake_normalized = np.reshape(fake, (1, -1, 1)) * n_bins
#     disc_output = discriminator(fake_normalized)
#     disc_output = expit(disc_output.numpy().squeeze())
#     print(f"Discriminator output {disc_output * 100:.2f}%")
#     h = np.arange(n_bins)
#     c = "red" if fake_mode == "gan" else "blue"
#     plt.bar(h, fake, width=1.0, alpha=0.5, linewidth=0.0, color=c)
#     if show_pdf:
#         plt.plot(h, d, lw=2, c="black")
# #     locs = [int((i) * (nbins - 1)) // 4 for i in range(5)]
# #     rfun = lambda x: round(10**x) if rounding == 1 else np.round(10**x, rounding)
# #     labs = [f"{rfun(u[j]):,}" for j in locs]
# #     plt.xticks(locs, labs, rotation=0, size=15)
# #     plt.yticks(size=15)
# #     plt.ylabel("counts", fontsize=16)
# #     plt.xlabel(dim.lower(), fontsize=16)
#     if savepath is not None:
#         plt.savefig(savepath, pad_inches=0.1, bbox_inches="tight")
#     plt.show()
    
#     return sim


# %%
# sim = plotfake(
#     "param",
#     sim_mode=1,
#     rounding=6,
#     n_bins=100,
#     seed=30,
#     nr=0.5,
#     show_pdf=True,
#     savepath="./examples/ganex_param_100_07.pdf")

# sim = plotfake(
#     "gan",
#     sim_mode=1,
#     rounding=6,
#     n_bins=100,
#     seed=30,
#     nr=0.5,
#     show_pdf=True,
#     savepath="./examples/ganex_gan_100_07.pdf")


# %%
# sim = plotfake(
#     "param",
#     sim_mode=2,
#     rounding=1,
#     n_bins=35,
#     seed=1896,
#     nr=0.5,
#     show_pdf=True,
#     savepath="./examples/ganex_param_peak_05.pdf")

# sim = plotfake(
#     "gan",
#     sim_mode=2,
#     rounding=1,
#     n_bins=35,
#     seed=1896,
#     nr=0.5,
#     show_pdf=True,
#     savepath="./examples/ganex_gan_peak_05.pdf")


# %%
# sim = plotfake(
#     "param",
#     sim_mode=1,
#     rounding=1,
#     n_bins=24,
#     seed=3698,
#     nr=0.1,
#     show_pdf=True,
#     savepath="./examples/ganex_param_24_01.pdf")

# sim = plotfake(
#     "gan",
#     sim_mode=1,
#     rounding=1,
#     n_bins=24,
#     seed=3698,
#     nr=0.1,
#     show_pdf=True,
#     savepath="./examples/ganex_gan_24_01.pdf")


# %%
# sim = plotfake(
#     "param",
#     sim_mode=1,
#     rounding=10,
#     n_bins=300,
#     seed=1796,
#     nr=0.5,
#     show_pdf=True,
#     savepath="./examples/ganex_param_300_05.pdf")

# sim = plotfake(
#     "gan",
#     sim_mode=1,
#     rounding=15,
#     n_bins=300,
#     seed=1796,
#     nr=0.5,
#     show_pdf=True,
#     savepath="./examples/ganex_gan_300_05.pdf")

# %% [markdown]
# ## 3. Denoise DoQ Densities

# %%
# obj_dim_pairs = []
# for obj in dat:
#     for dim in dat[obj]:
#         n_bins = len(dat[obj][dim]['counts'])
#         obj_dim_pairs.append((n_bins, obj, dim))
# obj_dim_pairs = sorted(obj_dim_pairs, key=lambda x: x[0], reverse=True)
# obj_dim_pairs = [(i, obj, dim) for i, (n_bins, obj, dim) in enumerate(obj_dim_pairs)]


# %%
# import pdb

def density_stats(dinput, x):
    nsamples = 10000
    qlevels = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
    d = dinput.astype(float) + 1e-6
    try:
        d /= d.sum()
    except:
        d = np.full(len(x), 1.0 / len(x))
        # pdb.set_trace()
    samples = np.random.choice(x, size=nsamples, p=d, replace=True)
    quants = np.quantile(samples, qlevels)
    mean = np.mean(samples)
    std = np.std(samples)
    mode = np.argmax(d)
    output = {
        'quantiles': quants.tolist(),
        'std': std,
        'mode': x[mode],
        'mean': mean,
        'median': quants[5],
        'iqr': quants[6] - quants[4]}
    return output


# call network in batches for efficiency    
def smooth_doq(batch_,
               peaks_prominence=0.001,
               peaks_min_rel_height=0.1,
               peaks_min_rel_distance=0.05,
               peaks_rel_threshold=0.001,
               tick_rotation=15):
    # 1. use smoother network
    #
    # input has 5 entries: obj, dim, loglower, logupper, counts
    #
    batch = [np.array(x[4], np.float32) for x in batch_]
    xs = [10.0 ** (0.5 * np.array(x[2]) + 0.5 * np.array(x[3]))
          for x in batch_]
    n_bins = [len(z) for z in batch]
    maxbins = max(n_bins)
    batchsize = len(batch)
    smoother_input = []
    for i in range(batchsize):
        z = np.zeros(maxbins, np.float32)
        l = (maxbins - len(batch[i])) // 2
        out = preprocess(batch[i])
        z[l:(l + n_bins[i])] = out.squeeze()
        smoother_input.append(np.reshape(z, (1, -1, 1)))
    smoother_input = np.concatenate(smoother_input, 0)
    density_, _ = smoother(smoother_input)
    density_ = density_.numpy().squeeze(-1)
    density_ = postprocess(density_)

    # 2. use context deconvolution network
    n_contexts = []
    peaks = []
    deconv_inputs = []
    for i in range(batchsize):
        d = density_[i]
        n_bins_i = n_bins[i]
        peaks_i = signal.find_peaks(
            d,
            height=peaks_min_rel_height / n_bins_i,
            distance=max(1, peaks_min_rel_distance * n_bins_i), 
            threshold=peaks_rel_threshold / n_bins_i,
            prominence=peaks_prominence)[0]
        n_contexts_i = len(peaks_i)
        if n_contexts_i == 0:
            peaks_i = np.array([np.argmax(d)])
            n_contexts_i = 1
        n_contexts.append(n_contexts_i)
        peaks.append(peaks_i)
        peaks_inputs = np.zeros((n_contexts_i, maxbins, 1), np.float32)
        for j, p in enumerate(peaks_i):
            peaks_inputs[j, p, 0] = 1.0
        counts_input = np.stack(n_contexts_i * [smoother_input[i]], 0)
        inputs = np.concatenate([counts_input, peaks_inputs], -1)
        deconv_inputs.append(inputs)
    deconv_inputs = np.concatenate(deconv_inputs, 0)   
    deconv_signals_, wts_nn_ = deconv(deconv_inputs)
    deconv_signals_ = deconv_signals_.numpy().squeeze(-1)
    deconv_signals_ = postprocess(deconv_signals_, threshold=0.1)
    wts_nn_ = wts_nn_.numpy().squeeze(-1)
    
    # 3. de-batch and get other metric
    start_at = 0
    wts_counts = []
    context_preds = []
    deconv_signals = []
    wts_nns = []
    densities = []
    entropies = []
    d_stats = []
    c_stats = []
    for i in range(batchsize):
        end_at = start_at + n_contexts[i]
        l = (maxbins - n_bins[i]) // 2
        w = softmax(wts_nn_[start_at:end_at])
        d = density_[i, l:(l + n_bins[i])]
        ds = deconv_signals_[start_at:end_at, l:(l + n_bins[i])]
#         entropy = - np.sum(d * np.log(d + 1e-6))
        deconv_signals.append(ds)
        densities.append(d)
#         entropies.append(float(entropy))
        wts_nns.append(w)
        try:
            d_stats.append(density_stats(d, xs[i]))
            c_stats_i = []
            for j in range(n_contexts[i]):
                c_stats_i.append(density_stats(ds[j], xs[i]))
            c_stats.append(c_stats_i)
        except:
            print("error while computing densities")
            print(f"obj: {batch_[i]['obj']}, dim: {batch_[i]['dim']}")
            
#         context_pred = ds.transpose() + 1e-10
#         row_sums = context_pred.sum(1, keepdims=True)
#         if np.any(row_sums == 0):
#             pdb.set_trace()
#         context_pred /= row_sums
        wts_count = ds.transpose() * np.expand_dims(batch[i], -1)
        wts_count = wts_count.sum(0)
        total = wts_count.sum()
        if total > 0:
            wts_count /= total
#         context_preds.append(context_pred)
        wts_counts.append(wts_count)
        start_at = end_at
            
    # 4. make batch outputs
    outputs_densities = []
    outputs_stats = []
    for i in range(batchsize):
        output = {
            'n_bins': n_bins[i],
            'density': densities[i].tolist(),
            'mode': float(xs[i][np.argmax(densities[i])]),
            'peaks': peaks[i].tolist(),
            'n_contexts': n_contexts[i],
#             'context_pred': context_preds[i].tolist(),
            'contexts': deconv_signals[i].tolist(),
            'wts_counts': wts_counts[i].tolist(),
            'wts_nn': wts_nns[i].tolist(),}
#             'entropy': entropies[i]}
        outputs_densities.append(output)
        output = {
            # 'peaks': peaks[i].tolist(),
            'n_contexts': n_contexts[i],
            'density': d_stats[i],
            'contexts': c_stats[i],
            'wts_counts': wts_counts[i].tolist(),
            'wts_nn': wts_nns[i].tolist(),}
#             'entropy': entropies[i]}
        outputs_stats.append(output)
    return outputs_densities, outputs_stats


# %%
BATCHSIZE = 64

# batch = []
# batch_num = 0
# for counter, obj, dim in tqdm(obj_dim_pairs):
#     batch.append(dat[obj][dim])
#     if (counter + 1) == len(obj_dim_pairs) or ((counter + 1) % BATCHSIZE == 0):
#         outputs_densities, outputs_stats = smooth_doq(batch)
#         for i in range(len(batch)):
#             outputs_densities[i]['counts'] = batch[i]['counts']
#             outputs_densities[i]['loglower'] = batch[i]['loglower']
#             outputs_densities[i]['logupper'] = batch[i]['logupper']
#             outputs_densities[i]['obj'] = batch[i]['obj']
#             outputs_stats[i]['obj'] = batch[i]['obj']
#             outputs_densities[i]['dim'] = batch[i]['dim']
#             outputs_stats[i]['dim'] = batch[i]['dim']
#         with open(f"./data/densities/batch_{batch_num + 1:04d}.json", "w") as io:
#             json.dump(outputs_densities, io)
#         with open(f"./data/statistics/batch_{batch_num + 1:04d}.json", "w") as io:
#             json.dump(outputs_stats, io)
#         batch = []
#         batch_num += 1


# %%

fdir = os.path.join(wdir, "./data/batches/")
files = glob.glob(fdir + "batch*.json")

dat = defaultdict(dict)
for batch_num, file in tqdm(enumerate(files)):
    # if batch_num < 9660:
    #     continue
    with open(file, "r") as f:
        batch = json.load(f)
        outputs_densities, outputs_stats = smooth_doq(batch)
        for i in range(len(batch)):
            outputs_densities[i]['counts'] = batch[i][4]
            outputs_densities[i]['logupper'] = batch[i][3]
            outputs_densities[i]['loglower'] = batch[i][2]
            outputs_densities[i]['dim'] = batch[i][1]
            outputs_stats[i]['dim'] = batch[i][1]
            outputs_densities[i]['obj'] = batch[i][0]
            outputs_stats[i]['obj'] = batch[i][0]
        with open(f"./data/densities/batch_{batch_num + 1:04d}.json", "w") as io:
            json.dump(outputs_densities, io)
        with open(f"./data/statistics/batch_{batch_num + 1:04d}.json", "w") as io:
            json.dump(outputs_stats, io)




# %%


