import os
from glob import glob
from datetime import datetime
import ujson
import numpy as np
import tensorflow as tf
from smoothdoq import gan, denoiser
from smoothdoq.layers import Ensemble
import matplotlib.pyplot as plt
from ml_logger import logbook as ml_logbook
import hydra
from smoothdoq import utils


sequence = tf.keras.preprocessing.sequence


def preprocess_batch(
    batch: list, logits=True, crop_frac=0.1, C=20, ls=1e-6,
) -> tuple:
    Dy = [b["pdf"] for b in batch]
    Dx = [b["sample"] for b in batch]
    x = sequence.pad_sequences(Dx, dtype="float", padding="post", maxlen=400)
    x = tf.constant(x, tf.float32)
    y = sequence.pad_sequences(Dy, dtype="float", padding="post", maxlen=400)
    mask = denoiser.compute_mask(Dx, maxlen=400)
    y = tf.constant(y, tf.float32)
    logits_y = tf.math.log(ls + y)
    logits_y -= tf.reduce_mean(logits_y, -1, keepdims=True)
    logits_y = tf.clip_by_value(logits_y, -C, C)
    logits_x = tf.math.log(ls + x)
    logits_x -= tf.reduce_mean(logits_x, -1, keepdims=True)
    logits_x = tf.clip_by_value(logits_x, -C, C)

    noise = tf.random.normal(logits_y.shape)

    return logits_y, logits_x, noise, x, y, mask


def kld(y, yhat, eps=1e-6):
    out = -y * (tf.math.log(yhat + eps) - tf.math.log(y + eps))
    loss = tf.reduce_sum(out, -1)
    return loss


def tvloss(z, k=1.0, order=2):
    pass


def plot_results(clean, recon, fake, title, file):
    N = len(clean)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(N), fake, width=1.1, color="blue", alpha=0.5, label="gen")
    ax.plot(clean, label="original", c="black")
    ax.plot(recon, label="recon", c="red")
    ax.legend()
    ax.set_title(title)
    fig.savefig(file, bbox_inches="tight", pad_inches=0)
    plt.close("all")


@hydra.main(config_name="6_free_gan.yml")
def main(cfg):
    utils.set_seed_everywhere(cfg.seed)
    plt.style.use("seaborn-colorblind")
    now = datetime.now().strftime("%d-%m-%Y/%H-%M-%S")
    logdir = "./" + (now if cfg.logdir is None else cfg.logdir)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(logdir + "/images", exist_ok=True)
    logbook_config = ml_logbook.make_config(
        logger_dir=logdir,
        write_to_console=False,
        tensorboard_config=dict(logdir=logdir),
    )
    logger = ml_logbook.LogBook(logbook_config)

    data = []
    wd = hydra.utils.get_original_cwd()
    for d in cfg.train_dirs:
        for fn in glob(f"{wd}/data/simulated/{d}/*.json"):
            with open(fn, "r") as io:
                batch = ujson.load(io)
                for record in batch:
                    data.append(record)

    # for d in cfg.ood_dirs:
    #     for fn in glob(f"{wd}/data/simulated/{d}/*.json"):
    #         with open(fn, "r") as io:
    #             batch = ujson.load(io)
    #             for record in batch:
    #                 ood_data.append(record)

    training_data = data
    # [data[i] for i in range(9, int(0.9 * len(data)))]
    # test_data = [data[i] for i in range(int(0.9 * len(data)), len(data))]
    # ood_data = [
    #     ood_data[i] for i in range(int(0.9 * len(ood_data)), len(ood_data))
    # ]

    def batches(data, batch_size):
        N = len(data) // batch_size
        for i in range(N):
            idx = range(i * batch_size, (i + 1) * batch_size)
            batch = [data[k] for k in idx]
            yield preprocess_batch(batch)

    nblocks = 8
    kernel_size = [3] + [3] * nblocks
    filters = 64
    dilation_rate = [1] + [1, 2, 4, 8, 1, 1, 1, 1]
    strides = [1] + [2, 1, 1, 1, 2, 1, 1, 2]
    generator_ensemble = [
        gan.FreeFormGAN(
            nblocks=nblocks,
            kernel_size=kernel_size,
            filters=filters,
            dilation_rate=dilation_rate,
        )
        for _ in range(cfg.ensemble_size)
    ]
    generator = Ensemble(generator_ensemble, stack_dim=1)
    discriminator = gan.Discriminator(
        nblocks=nblocks,
        kernel_size=kernel_size,
        filters=filters,
        dilation_rate=dilation_rate,
        strides=strides,
    )
    smoother = denoiser.BinDenoiser(
        nblocks=nblocks,
        kernel_size=kernel_size,
        filters=filters,
        dilation_rate=dilation_rate,
    )

    # lr assigned inside the loop
    gen_opt = tf.optimizers.Adam(1.0, clipnorm=cfg.clipnorm)
    disc_opt = tf.optimizers.Adam(1.0, clipnorm=cfg.clipnorm)
    smooth_opt = tf.optimizers.Adam(1.0, clipnorm=cfg.clipnorm)

    @tf.function
    def gen_train_step(
        logits_y,
        noise,
        x,
        y,
        mask,
        cycle_reg=1.0,
        identity_reg=1e-4,
        expl=0.1,
        train=True,
    ):
        B = logits_y.shape[0]
        with tf.GradientTape() as tape:
            fake = generator(logits_y, noise, training=True)
            disc_in = tf.reshape(fake, (-1, fake.shape[-1]))
            disc = discriminator(disc_in, training=False)
            disc = tf.reshape(disc, fake.shape[:2])
            # choose best one to tFFrain
            idx1 = tf.random.uniform(
                (B,), maxval=disc.shape[1], dtype=tf.int64
            )
            idx2 = tf.argmax(disc, 1)
            idx = tf.where(tf.random.uniform((B,)) < expl, idx1, idx2)
            idx_ = tf.stack([tf.range(B, dtype=tf.int64), idx], -1)
            disc = tf.gather_nd(disc, idx_)
            gen_loss = -tf.reduce_mean(disc)
            fake = tf.gather_nd(fake, idx_)
            logits_yhat = smoother(fake, training=False)
            recon = denoiser.masked_softmax(logits_yhat, mask)
            cycle_loss = tf.reduce_mean(kld(recon, y) + kld(y, recon))
            identity_loss = tf.reduce_mean(tf.math.abs(logits_y - fake))
            loss = (
                gen_loss
                + cycle_reg * cycle_loss
                + identity_reg * identity_loss
            )

        if train:
            grads = tape.gradient(loss, generator.trainable_variables)
            gen_opt.apply_gradients(zip(grads, generator.trainable_variables))

        return gen_loss, cycle_loss, fake, disc, recon, idx

    @tf.function
    def disc_train_step(logits_y, logits_x, fake, noise, train=True):
        with tf.GradientTape() as tape:
            disc_fake = discriminator(fake, training=True)
            disc_real = discriminator(logits_x, training=True)
            disc_fake = tf.reduce_mean(disc_fake)
            disc_real = tf.reduce_mean(disc_real)
            loss = disc_fake - disc_real

        if train:
            grads = tape.gradient(loss, discriminator.trainable_variables)
            disc_opt.apply_gradients(
                zip(grads, discriminator.trainable_variables)
            )

        return loss, disc_fake, disc_real

    @tf.function
    def smoother_train_step(fake, logits_x, y, mask, reg_wt, envs=1):
        with tf.GradientTape() as tape:
            w_dummy = tf.constant(1.0)
            logits_yhat = smoother(logits_x, training=True)
            fake_logits_yhat = smoother(fake, training=True)
            with tf.GradientTape() as tape2:
                tape2.watch(w_dummy)
                yhat = denoiser.masked_softmax(w_dummy * logits_yhat, mask, -1)
                fake_yhat = denoiser.masked_softmax(
                    w_dummy * fake_logits_yhat, mask, -1
                )
                loss_vec = kld(yhat, y) + kld(y, yhat)
                fake_loss_vec = kld(fake_yhat, y) + kld(y, fake_yhat)
                loss = tf.concat([loss_vec, fake_loss_vec], 0)
            # irm loss
            g, fake_g = tf.split(tape2.jacobian(loss, w_dummy), 2)
            irm_loss = 0.0
            for g_e in tf.split(g, envs) + [fake_g]:
                g1 = tf.reduce_mean(g_e[::2])
                g2 = tf.reduce_mean(g_e[1::2])
                irm_loss += g1 * g2 + 0.1 * (g1 + g2) ** 2
            # supervised smoother loss
            yhat = denoiser.masked_softmax(logits_yhat, mask, -1)
            loss_vec = kld(yhat, y) + kld(y, yhat)
            loss = tf.reduce_mean(loss_vec)
            total_loss = loss + reg_wt * irm_loss

        grads = tape.gradient(total_loss, smoother.trainable_variables)
        smooth_opt.apply_gradients(zip(grads, smoother.trainable_variables))

        return loss, irm_loss, yhat

    step = 0
    disc_loss_buff = []
    gen_loss_buff = []
    cycle_loss_buff = []
    smooth_loss_buff = []
    irm_loss_buff = []
    idx_pmin_buff = []
    epochs = cfg.epochs
    plot_every = cfg.plot_every
    batch_size = cfg.batch_size
    envs = cfg.batch_size // cfg.env_block_size
    cr = cfg.cycle_reg
    ir = cfg.identity_reg

    dfake_ma = 0.0
    dreal_ma = 0.0
    train_gen = True
    train_disc = True

    for e in range(epochs):
        rate = np.exp(-e * np.log(2) / cfg.half_lr)
        lr = cfg.min_lr + (cfg.init_lr - cfg.min_lr) * rate
        e0 = max(e - cfg.reg_warmup, 0)
        reg_wt = cfg.max_reg * (1.0 - np.exp(-e0 * np.log(2) / cfg.half_reg))
        disc_opt.lr.assign(lr)
        gen_opt.lr.assign(lr)
        smooth_opt.lr.assign(lr)
        rate = np.exp(-e * np.log(2) / cfg.half_expl)
        expl = cfg.min_expl + (cfg.init_expl - cfg.min_expl) * rate

        for j, batch_data in enumerate(batches(training_data, batch_size)):
            logits_y, logits_x, noise, x, y, mask = batch_data

            # update_gen = j % cfg.train_gen_every == 0
            gen_loss, cycle_loss, fake, disc, cycle, idx = gen_train_step(
                logits_y, noise, x, y, mask, cr, ir, expl, train_gen
            )
            freqs = np.zeros(cfg.ensemble_size)
            for k in idx:
                freqs[int(k)] += 1.0 / cfg.batch_size
            idx_pmin = min(freqs)
            idx_pmin_buff.append(float(idx_pmin))
            gen_loss_buff.append(float(gen_loss))
            cycle_loss_buff.append(float(cycle_loss))

            smooth_loss, irm_loss, yhat = smoother_train_step(
                fake, logits_x, y, mask, reg_wt, envs
            )
            smooth_loss_buff.append(float(smooth_loss))
            irm_loss_buff.append(float(irm_loss))

            disc_loss, disc_fake, disc_real = disc_train_step(
                logits_y, logits_x, fake, noise, train_disc
            )
            dfake_ma += cfg.band_lam * (disc_fake - dfake_ma)
            dreal_ma += cfg.band_lam * (disc_real - dreal_ma)
            disc_loss_buff.append(float(disc_loss))

            # whether train disc/gen or not in next rounds to guarantee eq
            if train_gen and dfake_ma - dreal_ma > cfg.band_lim:
                train_gen = False
                train_disc = True
                msg = f"gen off dreal={dreal_ma:.2f}, dfake={dfake_ma:.2f}"
                logger.write_message(msg)
            elif not train_gen and dfake_ma - dreal_ma < cfg.band_tgt:
                train_gen = True
                train_disc = True
                msg = f"gen on dreal={dreal_ma:.2f}, dfake={dfake_ma:.2f}"
                logger.write_message(msg)

            if train_disc and dfake_ma - dreal_ma < - cfg.band_lim:
                train_gen = True
                train_disc = False
                msg = f"disc off dreal={dreal_ma:.2f}, dfake={dfake_ma:.2f}"
                logger.write_message(msg)
            elif not train_disc and dfake_ma - dreal_ma > - cfg.band_tgt:
                train_gen = True
                train_disc = True
                msg = f"disc on dreal={dreal_ma:.2f}, dfake={dfake_ma:.2f}"
                logger.write_message(msg)

            if step % (plot_every // cfg.train_gen_every) == 0:
                fn = f"{logdir}/images/{step:05d}.png"
                pfake0 = denoiser.masked_softmax(fake[0], mask[0])
                title = f"D = {float(disc[0]):.2f}"
                plot_results(
                    *[u.numpy() for u in (y[0], yhat[0], pfake0)], title, fn
                )

            step += 1

        gen_loss = np.mean(gen_loss_buff)
        gen_loss_buff.clear()
        disc_loss = np.mean(disc_loss_buff)
        disc_loss_buff.clear()
        cycle_loss = np.mean(cycle_loss_buff)
        cycle_loss_buff.clear()
        smooth_loss = np.mean(smooth_loss_buff)
        smooth_loss_buff.clear()
        irm_loss = np.mean(irm_loss_buff)
        irm_loss_buff.clear()
        idx_pmin = np.mean(idx_pmin_buff)
        idx_pmin_buff.clear()
        logger.write_metric({"global_step": e, "loss/gen": gen_loss})
        logger.write_metric({"global_step": e, "loss/disc": disc_loss})
        logger.write_metric({"global_step": e, "loss/cycle": cycle_loss})
        logger.write_metric({"global_step": e, "loss/smooth": smooth_loss})
        logger.write_metric({"global_step": e, "loss/irm": irm_loss})
        logger.write_metric({"global_step": e, "info/lr": lr})
        logger.write_metric({"global_step": e, "info/idx_pmin": idx_pmin})
        logger.write_metric({"global_step": e, "info/expl": expl})

    step += 1


if __name__ == "__main__":
    main()
