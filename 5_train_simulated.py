import sys
import os
import glob
import ml_logger
from datetime import datetime
from collections import deque
from collections import defaultdict
import ujson
import numpy as np
import tensorflow as tf
from smoothdoq import denoiser, noise, binned
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from ml_logger import logbook as ml_logbook

plt.style.use("seaborn-colorblind")
now = datetime.now().strftime("%d-%m-%Y/%H-%M-%S")


def preprocess_batch(
    batch: list, logits=True, crop_frac=0.1, eps=1e-12, training=True
):
    Dx = []
    Dy = []
    max_logits = -tf.math.log(eps)

    # preprocess logits of true pdf
    for b in batch:
        if training:
            # use random cropping and reflection
            wlen = b["window_len"]
            ws = b["window_start"]
            N = b["n_bins"]
            new_ws = np.random.randint(N - wlen)
            x = np.array(b["sample"], int)
            y = np.array(b["pdf"])
            newx = np.zeros(N, int)
            newy = np.zeros(N)
            rev = np.random.rand() < 0.5
            if np.random.rand() < 0.5:
                idx = np.arange(ws, ws + wlen)
            else:
                idx = np.arange(ws + wlen - 1, ws - 1, -1)
            newx[new_ws : new_ws + wlen] = x[idx]
            newy[new_ws : new_ws + wlen] = y[idx]
            Dx.append(newx.tolist())
            Dy.append(newy.tolist())
        else:
            Dx.append(b["sample"])
            Dy.append(b["pdf"])

    x = pad_sequences(Dx, dtype="float", padding="post", maxlen=400)
    y = pad_sequences(Dy, dtype="float", padding="post", maxlen=400)
    mask = denoiser.compute_mask(Dx, maxlen=400)
    y = tf.constant(y, tf.float32)
    x = tf.constant(x, tf.float32)

    logits_x = tf.math.log(eps + x)
    logits_x -= tf.reduce_mean(logits_x, -1, keepdims=True)
    logits_x = tf.clip_by_value(logits_x, -max_logits, max_logits)
    logits_y = tf.math.log(eps + y)
    logits_y -= tf.reduce_mean(logits_y, -1, keepdims=True)
    logits_y = tf.clip_by_value(logits_y, -max_logits, max_logits)

    return logits_x, x, logits_y, y, mask


def kld(y, yhat):
    eps = 1e-14
    out = -y * tf.math.log((yhat + eps) / (y + eps))
    return tf.reduce_mean(tf.reduce_sum(out, -1))


def tvloss(z, lam=1e-3):
    d = z[:, 1:] - z[:, :-1]
    tv = tf.reduce_sum(d ** 2, -1)
    loss = lam * tf.reduce_mean(tv)
    return loss


def get_lr(epoch, init_lr):
    return max(1e-6, init_lr * np.exp(-epoch * np.log(2) / 100))


def kde(x, mask, bw=None, bw_factor=1.0):
    out = np.zeros(len(x))
    N = x.sum()
    xtilde = np.arange(len(x)).astype(float)

    if bw is None:
        # estimate mean and stdev
        mean = np.sum(xtilde * x / N)
        sumsq = np.sum(xtilde ** 2 * x / N)
        std = np.sqrt(sumsq - mean ** 2)
        h = 1.06 * std * N ** (-0.2)
        bw = bw_factor * 1.0 / h

    for xi, n in enumerate(x):
        wt = n / N
        d = np.sqrt(bw / np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * (bw * (xi - xtilde)) ** 2
        )
        out += wt * d

    out *= mask
    out /= out.sum()

    return out


def plot_results(x, y, yhat, mask, step):
    x = x / sum(x)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(x)), x, width=1.1, color="blue", alpha=0.5)
    ax.plot(yhat, label="fitted", c="red")
    ax.plot(y, label="true", c="black")
    y_kde = kde(x, mask)
    y_kde2 = kde(x, mask, bw_factor=5.0)
    ax.plot(y_kde, label="kde-bw*", c="magenta")
    ax.plot(y_kde2, label="kde-5bw*", c="darkgreen")
    ax.legend()
    fig.savefig(
        f"{logdir}/images/{step:05d}.pdf", bbox_inches="tight", pad_inches=0,
    )
    plt.close("all")


if __name__ == "__main__":

    logdir = "./tflogs/" + now
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(logdir + "/images", exist_ok=True)
    logbook_config = ml_logbook.make_config(
        logger_dir=logdir, tensorboard_config=dict(logdir=logdir)
    )
    logger = ml_logbook.LogBook(logbook_config)

    data = []
    for fn in glob.glob("./data/simulated/normal/*.json"):
        with open(fn, "r") as io:
            batch = ujson.load(io)
            for record in batch:
                data.append(record)
    training_data = [data[i] for i in range(9, int(0.9 * len(data)))]
    test_data = [data[i] for i in range(int(0.9 * len(data)), len(data))]

    def batches(data, batch_size, training=True):
        N = len(data) // batch_size
        for i in range(N):
            idx = range(i * batch_size, (i + 1) * batch_size)
            batch = [data[k] for k in idx]
            yield preprocess_batch(batch, training=training)

    nblocks = 8
    kernel_size = [3] + [3] * nblocks
    filters = 32
    dilation_rate = [1] + [1, 2, 4, 8, 1, 1, 1, 1]

    model = denoiser.BinDenoiser(
        nblocks=nblocks,
        kernel_size=kernel_size,
        filters=filters,
        dilation_rate=dilation_rate,
        dropout=0.5,
    )

    init_lr = 0.01
    optimizer = tf.optimizers.Adam(init_lr)

    @tf.function  # (experimental_relax_shapes=True)
    def train_step(logits_x, y, mask, training=True):
        with tf.GradientTape() as tape:
            logits_yhat = model(logits_x, training=training, mask=mask)
            yhat = denoiser.masked_softmax(logits_yhat, mask, -1)
            loss = kld(yhat, y) + kld(y, yhat)
            reg = tvloss(logits_yhat, lam=1e-3)
            total_loss = reg + loss

        if training:
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )

        return loss, reg, yhat

    @tf.function  # (experimental_relax_shapes=True)
    def test_step(logits_x, y, mask):
        logits_yhat = model(logits_x, training=False, mask=mask)
        yhat = denoiser.masked_softmax(logits_yhat, mask, -1)
        loss = kld(yhat, y) + kld(y, yhat)
        return loss, yhat

    step = 0
    plot_every = 1_000
    loss_buff = []
    reg_buff = []
    test_loss_buff = []
    epochs = 20_000
    batch_size = 32

    for e in range(epochs):
        lr = get_lr(e, init_lr)
        optimizer.lr.assign(lr)

        for test_batch_data in batches(test_data, batch_size):
            logits_x, x, logits_y, y, mask = test_batch_data
            loss, yhat = test_step(logits_x, y, mask)
            test_loss_buff.append(float(loss))

        for batch_data in batches(training_data, batch_size):
            logits_x, x, logits_y, y, mask = batch_data
            loss, reg, yhat = train_step(logits_x, y, mask)
            loss_buff.append(float(loss))
            reg_buff.append(float(reg))

            if step % plot_every == 0:
                plot_results(
                    x[0].numpy(),
                    y[0].numpy(),
                    yhat[0].numpy(),
                    mask[0].numpy(),
                    step,
                )

            step += 1

        logger.write_metric(
            {"global_step": e, "loss/train": np.mean(loss_buff)}
        )
        logger.write_metric(
            {"global_step": e, "loss/test": np.mean(test_loss_buff)}
        )
        logger.write_metric({"global_step": e, "loss/reg": np.mean(reg_buff)})
        loss_buff = []
        test_loss_buff = []
        reg_buff = []
        logger.write_metric({"global_step": e, "step": step})
        logger.write_metric({"global_step": e, "lr": lr})

        model.save_weights(f"{logdir}/ckpt.h5")

    step += 1
