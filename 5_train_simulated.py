import os
from glob import glob
from datetime import datetime
import ujson
import numpy as np
import tensorflow as tf
from smoothdoq import denoiser
import matplotlib.pyplot as plt
from ml_logger import logbook as ml_logbook
import hydra
from smoothdoq import utils


pad_sequences = tf.keras.preprocessing.sequence


def preprocess_batch(
    batch: list, logits=True, crop_frac=0.1, C=20, training=True, ls=1e-6,
):
    Dx = []
    Dy = []

    # preprocess logits of true pdf
    for b in batch:
        if training:
            # use random cropping and reflection
            # wlen = b["window_len"]
            # ws = b["window_start"]
            # N = b["n_bins"]
            # new_ws = np.random.randint(N - wlen)
            # x = np.array(b["sample"], int)
            # y = np.array(b["pdf"])
            # newx = np.zeros(N, int)
            # newy = np.zeros(N)
            if np.random.rand() < 0.5:
                # idx = np.arange(ws, ws + wlen)
                newx = b["sample"]
                newy = b["pdf"]
            else:
                # idx = np.arange(ws + wlen - 1, ws - 1, -1)
                newx = list(reversed(b["sample"]))
                newy = list(reversed(b["pdf"]))
            # newx[new_ws : new_ws + wlen] = x[idx]
            # newy[new_ws : new_ws + wlen] = y[idx]
            # newx = newx.tolist()
            # newy = newy.tolist()
            Dx.append(newx)
            Dy.append(newy)
        else:
            Dx.append(b["sample"])
            Dy.append(b["pdf"])

    x = pad_sequences(Dx, dtype="float", padding="post", maxlen=400)
    y = pad_sequences(Dy, dtype="float", padding="post", maxlen=400)
    mask = denoiser.compute_mask(Dx, maxlen=400)
    y = tf.constant(y, tf.float32)
    x = tf.constant(x, tf.float32)

    logits_x = tf.math.log(ls + x)
    logits_x -= tf.reduce_mean(logits_x, -1, keepdims=True)
    logits_x = tf.clip_by_value(logits_x, -C, C)
    logits_y = tf.math.log(ls + y)
    logits_y -= tf.reduce_mean(logits_y, -1, keepdims=True)
    logits_y = tf.clip_by_value(logits_y, -C, C)

    return logits_x, x, logits_y, y, mask


def kld(y, yhat, eps=1e-6):
    out = - y * (tf.math.log(yhat + eps) - tf.math.log(y + eps))
    loss = tf.reduce_sum(out, -1)
    return loss


def tvloss(z, k=1.0, order=2):
    d = z
    for i in range(order + 1):
        d = d[:, 1:] - d[:, :-1]
    d = tf.math.abs(d)
    tv = tf.where(d < k, 0.5 * d ** 2, k * (d - 0.5 * k))
    loss = tf.reduce_mean(tf.reduce_sum(tv, -1))
    return loss


def quantile(x, q):
    assert 0 <= q <= 1
    z = np.cumsum(x)
    z /= z[-1]
    for i, val in enumerate(z):
        if val > q:
            return i


def kde(x, mask, bw=None, bw_factor=1.0, silverman=False):
    out = np.zeros(len(x))
    N = x.sum()
    xtilde = np.arange(len(x)).astype(float)

    if bw is None:
        # estimate mean and stdev
        mean = np.sum(xtilde * x / N)
        sumsq = np.sum(xtilde ** 2 * x / N)
        std = np.sqrt(sumsq - mean ** 2)
        if not silverman:
            h = 1.06 * std * N ** (-0.2)
        else:
            iqr = quantile(x, 0.75) - quantile(x, 0.25)
            h = 0.9 * min(std, (iqr + 0.01) / 1.34) * N ** (-0.2)
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


def plot_results(x, y, yhat, mask, file):
    N = sum(x)
    if N == 0:
        raise Exception("x is all zeros")
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
    fig.savefig(file, bbox_inches="tight", pad_inches=0)
    plt.close("all")
    return y_kde, y_kde2


@hydra.main(config_name="5_train_simulated.yml")
def main(cfg):
    utils.set_seed_everywhere(cfg.seed)
    plt.style.use("seaborn-colorblind")
    now = datetime.now().strftime("%d-%m-%Y/%H-%M-%S")
    logdir = "./tflogs/" + (now if cfg.logdir is None else cfg.logdir)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(logdir + "/images", exist_ok=True)
    logbook_config = ml_logbook.make_config(
        logger_dir=logdir,
        write_to_console=False,
        tensorboard_config=dict(logdir=logdir),
    )
    logger = ml_logbook.LogBook(logbook_config)

    data = []
    ood_data = []
    for d in cfg.train_dirs:
        for fn in glob(f"./data/simulated/{d}/*.json"):
            with open(fn, "r") as io:
                batch = ujson.load(io)
                for record in batch:
                    data.append(record)

    for d in cfg.ood_dirs:
        for fn in glob(f"./data/simulated/{d}/*.json"):
            with open(fn, "r") as io:
                batch = ujson.load(io)
                for record in batch:
                    ood_data.append(record)

    training_data = [data[i] for i in range(9, int(0.9 * len(data)))]
    test_data = [data[i] for i in range(int(0.9 * len(data)), len(data))]
    ood_data = [
        ood_data[i] for i in range(int(0.9 * len(ood_data)), len(ood_data))
    ]

    def batches(data, batch_size, training=True):
        N = len(data) // batch_size
        for i in range(N):
            idx = range(i * batch_size, (i + 1) * batch_size)
            batch = [data[k] for k in idx]
            yield preprocess_batch(batch, training=training)

    nblocks = 8
    kernel_size = [3] + [3] * nblocks
    filters = 64
    dilation_rate = [1] + [1, 2, 4, 8, 1, 1, 1, 1]

    model = denoiser.BinDenoiser(
        nblocks=nblocks,
        kernel_size=kernel_size,
        filters=filters,
        dilation_rate=dilation_rate,
        dropout=0.0,
    )

    init_lr = cfg.init_lr
    optimizer = tf.optimizers.Adam(init_lr, clipnorm=cfg.clipnorm)

    @tf.function
    def train_step(
        logits_x,
        y,
        mask,
        reg_wt,
        irm_rel_wt=1.0,
        tv_rel_wt=1.0,
        envs=1,
        training=True,
    ):
        with tf.GradientTape() as tape:
            w_dummy = tf.constant(1.0)
            logits_yhat = model(logits_x, training=training, mask=mask)
            with tf.GradientTape() as tape2:
                tape2.watch(w_dummy)
                yhat = denoiser.masked_softmax(w_dummy * logits_yhat, mask, -1)
                loss_vec = kld(yhat, y) + kld(y, yhat)
            # irm loss
            g = tape2.jacobian(loss_vec, w_dummy)
            irm_loss = 0.0
            for g_e in tf.split(g, envs):
                g1 = tf.reduce_mean(g_e[::2])
                g2 = tf.reduce_mean(g_e[1::2])
                irm_loss += g1 * g2 + 0.25 * (g1 + g2) ** 2
            tv_reg = tvloss(logits_yhat, k=cfg.huber_k, order=cfg.tv_order)
            logits_yhat = model(logits_x, training=training, mask=mask)
            yhat = denoiser.masked_softmax(logits_yhat, mask, -1)
            loss_vec = kld(yhat, y) + kld(y, yhat)
            loss = tf.reduce_mean(loss_vec)
            total_loss = loss + reg_wt * (
                tv_rel_wt * tv_reg + irm_rel_wt * irm_loss
            )

        if training:
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )

        return loss, tv_reg, irm_loss, yhat

    @tf.function
    def test_step(logits_x, y, mask):
        logits_yhat = model(logits_x, training=False, mask=mask)
        yhat = denoiser.masked_softmax(logits_yhat, mask, -1)
        loss = kld(yhat, y) + kld(y, yhat)
        loss = tf.reduce_mean(loss)
        return loss, yhat

    step = 0
    loss_buff = []
    reg_buff = []
    test_loss_buff = []
    ood_buff, ood_kde_buff, ood_kde2_buff, ood_kdeS_buff = [], [], [], []
    irm_buff = []
    epochs = cfg.epochs
    plot_every = cfg.plot_every
    batch_size = cfg.batch_size
    envs_per_batch = cfg.batch_size // cfg.env_block_size

    for e in range(epochs):
        rate = np.exp(-e * np.log(2) / cfg.half_lr)
        lr = cfg.min_lr + (cfg.init_lr - cfg.min_lr) * rate
        e0 = max(e - cfg.reg_warmup, 0)
        reg_wt = cfg.max_reg * (1.0 - np.exp(- e0 * np.log(2) / cfg.half_reg))
        optimizer.lr.assign(lr)

        for test_batch_data in batches(4 * test_data, batch_size, False):
            logits_x, x, logits_y, y, mask = test_batch_data
            loss, yhat = test_step(logits_x, y, mask)
            test_loss_buff.append(float(loss))

        for batch_data in batches(training_data, batch_size, False):
            logits_x, x, logits_y, y, mask = batch_data
            loss, reg, irm_loss, yhat = train_step(
                logits_x,
                y,
                mask,
                reg_wt,
                cfg.irm_rel_wt,
                cfg.tv_rel_wt,
                envs_per_batch,
            )
            loss_buff.append(float(loss))
            reg_buff.append(float(reg))
            irm_buff.append(float(irm_loss))

            step += 1

            if step % plot_every == 0:
                fn = f"{logdir}/images/{step:05d}.png"
                plot_results(*[v[0].numpy() for v in (x, y, yhat, mask)], fn)

        ood_plot_example = np.random.randint(len(ood_data))
        j = 0

        for batch_data in batches(ood_data, 4 * batch_size, False):
            logits_x, x, logits_y, y, mask = batch_data
            loss, yhat = test_step(logits_x, y, mask)
            ood_buff.append(float(loss))

            if e == 0:
                y_kde = [
                    kde(xi.numpy(), mi.numpy())
                    for xi, mi in zip(x, mask)
                ]
                y_kde2 = [
                    kde(xi.numpy(), mi.numpy(), bw_factor=5.0)
                    for xi, mi in zip(x, mask)
                ]
                y_kdeS = [
                    kde(xi.numpy(), mi.numpy(), bw_factor=5.0, silverman=True)
                    for xi, mi in zip(x, mask)
                ]
                y_kde = tf.constant(np.stack(y_kde), tf.float32)
                y_kde2 = tf.constant(np.stack(y_kde2), tf.float32)
                y_kdeS = tf.constant(np.stack(y_kdeS), tf.float32)
                kde_loss = tf.reduce_mean(kld(y_kde, y) + kld(y, y_kde))
                kde2_loss = tf.reduce_mean(kld(y_kde2, y) + kld(y, y_kde2))
                kdeS_loss = tf.reduce_mean(kld(y_kdeS, y) + kld(y, y_kdeS))
                ood_kde_buff.append(float(kde_loss))
                ood_kde2_buff.append(float(kde2_loss))
                ood_kdeS_buff.append(float(kdeS_loss))

            for r in range(x.shape[0]):
                if j == ood_plot_example:  # print at random
                    fn = f"{logdir}/images/ood_{step:05d}.png"
                    plot_results(
                        *[v[r].numpy() for v in (x, y, yhat, mask)], fn
                    )
                j += 1

        train_loss = np.mean(loss_buff)
        tv_loss = np.mean(reg_buff)
        irm_loss = np.mean(irm_buff)
        logger.write_metric({"global_step": e, "loss/train": train_loss})
        logger.write_metric(
            {"global_step": e, "loss/test": np.mean(test_loss_buff)}
        )
        logger.write_metric({"global_step": e, "loss/tv": tv_loss})
        logger.write_metric({"global_step": e, "loss/irm": irm_loss})
        logger.write_metric({"global_step": e, "ood/sdoq": np.mean(ood_buff)})
        logger.write_metric(
            {"global_step": e, "ood/kde": np.mean(ood_kde_buff)}
        )
        logger.write_metric(
            {"global_step": e, "ood/kde2": np.mean(ood_kde2_buff)}
        )
        logger.write_metric(
            {"global_step": e, "ood/kdeS": np.mean(ood_kdeS_buff)}
        )
        logger.write_metric({"global_step": e, "info/step": step})
        logger.write_metric({"global_step": e, "info/lr": lr})
        logger.write_metric({"global_step": e, "info/reg_weight": reg_wt})
        reg_rel_wt = reg_wt * (tv_loss + irm_loss) / train_loss
        logger.write_metric({"global_step": e, "info/reg_rel_wt": reg_rel_wt})
        logger.write_metric(
            {"global_step": e, "info/irm_tv_ratio": tv_loss / irm_loss}
        )
        logger.write_metric(
            {"global_step": e, "info/reg_rel_wt": reg_wt / train_loss}
        )

        loss_buff, test_loss_buff = [], []
        reg_buff, irm_buff = [], []
        ood_buff = []

        model.save_weights(f"{logdir}/ckpt.h5")
        if step + 1 % cfg.ckpt_every == 0:
            model.save_weights(f"{logdir}/ckpt_{step + 1}.h5")

    step += 1


if __name__ == "__main__":
    main()
