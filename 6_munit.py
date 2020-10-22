import os
from glob import glob
from datetime import datetime
import ujson
import numpy as np
import tensorflow as tf
from smoothdoq import mgan, denoiser
from smoothdoq.denoiser import masked_softmax
import matplotlib.pyplot as plt
from ml_logger import logbook as ml_logbook
import hydra
from smoothdoq import utils

# from tensorflow_addons.optimizers import AdamW

sequence = tf.keras.preprocessing.sequence


class Replay:
    def __init__(self, capacity: int, dim: int, block: int) -> None:
        self._next_idx = 0
        self._dim = dim
        self._block = block
        self._N = capacity
        self._storage1 = np.zeros((capacity, dim), dtype=np.float32)
        self._storage2 = np.zeros((capacity, dim), dtype=np.float32)
        self._storage3 = np.zeros((capacity, dim), dtype=np.bool)

    def sample(self, blocks) -> np.ndarray:
        ix_b = np.random.randint(0, self._next_idx // self._block, size=blocks)
        ix_b = (ix_b * self._block) % self._N
        ix = []
        for i in ix_b:
            for j in range(self._block):
                ix.append(i + j)
        return self._storage1[ix], self._storage2[ix], self._storage3[ix]

    def add(
        self, block1: np.ndarray, block2: np.ndarray, block3: np.ndarray
    ) -> None:
        B = block1.shape[0]
        ix = [(self._next_idx + i) % self._N for i in range(B)]
        self._storage1[ix] = block1
        self._storage2[ix] = block2
        self._storage3[ix] = block3
        self._next_idx += B


def preprocess_batch(
    batch_train: list,
    batch_ood: list,
    logits=True,
    crop_frac=0.1,
    C=20,
    ls=1e-6,
    ebsize=1,
) -> tuple:
    Dy = [b["pdf"] for b in batch_train]
    Dx = [b["sample"] for b in batch_ood]
    x = sequence.pad_sequences(Dx, dtype="float", padding="post", maxlen=400)
    x = tf.constant(x, tf.float32)
    x /= tf.reduce_sum(x, -1, keepdims=True)
    y = sequence.pad_sequences(Dy, dtype="float", padding="post", maxlen=400)
    mask_y = denoiser.compute_mask(Dy, maxlen=400)
    mask_x = denoiser.compute_mask(Dx, maxlen=400)
    y = tf.constant(y, tf.float32)
    logits_y = tf.math.log(ls + y)
    logits_y -= tf.reduce_mean(logits_y, -1, keepdims=True)
    logits_y = tf.expand_dims(logits_y, -1)
    logits_y = tf.clip_by_value(logits_y, -C, C)
    logits_x = tf.math.log(ls + x)
    logits_x -= tf.reduce_mean(logits_x, -1, keepdims=True)
    logits_x = tf.expand_dims(logits_x, -1)
    logits_x = tf.clip_by_value(logits_x, -C, C)

    # noise = tf.random.uniform(logits_y.shape)
    noise = None
    alpha = None
    # envs = logits_x.shape[0] // ebsize
    # alpha = tf.repeat(
    #     tf.random.uniform(shape=(envs,), minval=0.00, maxval=1.0), ebsize
    # )

    return logits_y, logits_x, noise, alpha, x, y, mask_y, mask_x


def kld(y, yhat, eps=1e-6):
    out = -(y + eps) * tf.math.log((yhat + eps) / (y + eps))
    loss = tf.reduce_sum(out, -1)
    return loss


def js(y, yhat, eps=1e-6):
    return kld(y, yhat, eps) + kld(yhat, y, eps)


def tvloss(z, k=1.0, order=2):
    pass


def huber(x, k=1.0):
    d = tf.math.abs(x)
    return tf.where(d < k, 0.5 * d ** 2, k * (d - 0.5 * k))


def w1(y, yhat, mask, k):
    y_cum = tf.math.cumsum(y, -1)
    yhat_cum = tf.math.cumsum(yhat, -1)
    delta = huber(y_cum - yhat_cum, 0.01)
    B = tf.reduce_sum(mask, -1)  # num bins
    return tf.math.reduce_sum(delta, -1) / B


def to_logits(x, eps=1e-6, C=20, axis=-1):
    z = tf.math.log(x + eps)
    z_c = tf.clip_by_value(tf.stop_gradient(z), -C, C)
    M = tf.reduce_mean(z_c, axis, keepdims=True)
    return z - M


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


def plot_translate_results(clean, recon, fake, real, recon_real, title, file):
    N = len(clean)
    xr = range(N)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].bar(range(N), fake, width=1.1, color="blue", alpha=0.5, label="gen")
    ax[0].plot(clean, label="original", c="black")
    ax[0].plot(recon, label="recon", c="red")
    ax[0].legend()
    ax[0].set_title(title)
    ax[1].bar(
        xr, real, width=1.1, color="blue", alpha=0.5, label="style"
    )
    ax[1].legend()
    ax[2].bar(
        xr, recon_real, width=1.1, color="red", alpha=0.5, label="recon_style"
    )
    ax[2].legend()
    fig.savefig(file, bbox_inches="tight", pad_inches=0)
    plt.close("all")


@hydra.main(config_name="6_munit.yml")
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

    ood_data = []
    for d in cfg.ood_dirs:
        for fn in glob(f"{wd}/data/simulated/{d}/*.json"):
            with open(fn, "r") as io:
                batch = ujson.load(io)
                for record in batch:
                    ood_data.append(record)

    training_data = [data[i] for i in range(9, int(0.9 * len(data)))]
    test_data = [data[i] for i in range(int(0.9 * len(data)), len(data))]
    ood_data = [ood_data[i] for i in range(int(0.9 * len(ood_data)))]

    def batches(train_data, ood_data, batch_size):
        N = len(train_data) // batch_size
        N2 = len(ood_data) // batch_size
        for i in range(min(N, N2)):
            idx = range(i * batch_size, (i + 1) * batch_size)
            batch_train = [train_data[k] for k in idx]
            batch_ood = [ood_data[k] for k in idx]
            yield preprocess_batch(
                batch_train, batch_ood, ebsize=cfg.env_block_size
            )

    # content encoders
    content_enc_noi = mgan.ConvPipe(
        kernel_size=[7, 5, 3, 3],
        filters=[16, 32, 64, 64],
        strides=[2, 2, 2, 1],
        dilation=[1, 1, 1, 1],
        activation="leaky_relu",
        norm="in",
        residual=[False, False, False, True],
    )
    content_enc_smo = mgan.ConvPipe(
        kernel_size=[7, 5, 3, 3],
        filters=[16, 32, 64, 64],
        strides=[2, 2, 2, 1],
        dilation=[1, 1, 1, 1],
        activation="leaky_relu",
        norm="in",
        residual=[False, False, False, True],
    )

    # style encoders
    style_dim = 64
    style_enc_noi = mgan.ConvPipe(
        kernel_size=[7, 5, 5],
        filters=[16, 32, style_dim],
        strides=[2, 2, 2],
        dilation=[1, 1, 1],
        activation="leaky_relu",
        norm="none",
        residual=[False, False, False],
        pool=True,
        pool_alpha=1.0,
    )
    style_enc_smo = mgan.ConvPipe(
        kernel_size=[7, 5, 5],
        filters=[16, 32, style_dim],
        strides=[2, 2, 2],
        dilation=[1, 1, 1],
        activation="leaky_relu",
        norm="none",
        residual=[False, False, False],
        pool=True,
        pool_alpha=1.0,
    )

    # decoders
    dec_filters = [64, 64, 32, 16, 1]
    decoder_noi = mgan.ConvPipe(
        kernel_size=[3, 3, 5, 5, 5],
        filters=dec_filters,
        strides=[1, 1, 2, 2, 2],
        dilation=[1, 1, 1, 1, 1],
        residual=[True, True, False, False, False],
        upsample=True,
        activation="leaky_relu",
        norm="adain",
    )
    decoder_smo = mgan.ConvPipe(
        kernel_size=[3, 3, 5, 5, 5],
        filters=dec_filters,
        strides=[1, 1, 2, 2, 2],
        dilation=[1, 1, 1, 1, 1],
        residual=[True, True, False, False, False],
        upsample=True,
        activation="leaky_relu",
        norm="adain",
    )

    # prior transform / adain weights
    mlp_noi = mgan.MLP(
        output_dim=2 * sum(dec_filters),  # 2 x dec filters
        inner_dim=128,
        n_blk=3,
        norm="none",
        activation="relu",
    )
    mlp_smo = mgan.MLP(
        output_dim=2 * sum(dec_filters),
        inner_dim=128,
        n_blk=3,
        norm="none",
        activation="relu",
    )

    # discriminator
    disc_noi = mgan.ConvPipe(
        kernel_size=[7, 5, 5, 3],
        filters=[32, 16, 16, 1],
        strides=[2, 2, 2, 1],
        dilation=[1, 1, 1, 1],
        residual=[False, False, False, False],
        activation="leaky_relu",
        norm="in",
        pool=True,
    )
    disc_smo = mgan.ConvPipe(
        kernel_size=[7, 5, 5, 3],
        filters=[32, 16, 16, 1],
        strides=[2, 2, 2, 1],
        dilation=[1, 1, 1, 1],
        residual=[False, False, False, False],
        activation="leaky_relu",
        norm="in",
        pool=True,
    )

    # for comparison with previous approach
    # directly trained a denoiser
    smoother = denoiser.BinDenoiser(
        nblocks=2,
        kernel_size=[21, 3, 7],
        filters=128,
        dilation_rate=[1, 8, 1],
    )

    if cfg.use_replay:
        buffer = Replay(cfg.replay_capacity, 400, cfg.env_block_size)

    # lr assigned inside the loop
    gen_opt = tf.optimizers.Adam(1.0, beta_1=0.5, clipnorm=cfg.clipnorm)
    disc_opt = tf.optimizers.Adam(1.0, beta_1=0.01, clipnorm=cfg.clipnorm)
    smooth_opt = tf.optimizers.Adam(1.0, beta_1=0.01, clipnorm=cfg.clipnorm)

    @tf.function
    def identity_train_step(y, logits_y, mask_y, x, logits_x, mask_x):
        with tf.GradientTape() as tape:
            # 1. obtain the style and content codes
            content_noi = content_enc_noi(logits_x, training=True)
            content_smo = content_enc_smo(logits_y, training=True)
            style_noi = style_enc_noi(logits_x, training=True)
            style_smo = style_enc_noi(logits_y, training=True)

            # 2. obtain reconstructions
            weights_noi = mlp_noi(style_noi)
            weights_smo = mlp_smo(style_smo)
            recon_noi = decoder_noi(content_noi, weights_noi, training=True)
            recon_smo = decoder_smo(content_smo, weights_smo, training=True)

            # 3. cross domain decode
            noi2smo = decoder_smo(content_noi, weights_smo, training=True)
            smo2noi = decoder_noi(content_smo, weights_noi, training=True)
            content_noi2smo = content_enc_smo(noi2smo, training=True)
            content_smo2noi = content_enc_noi(smo2noi, training=True)
            style_noi2smo = style_enc_smo(noi2smo, training=True)
            style_smo2noi = style_enc_noi(smo2noi, training=True)

            # 4. compute losses
            loss_recon_noi = tf.reduce_mean(huber(recon_noi - logits_x), 1)
            recon_smo_prob = masked_softmax(tf.squeeze(recon_smo, -1), mask_y)
            loss_recon_smo = tf.reduce_mean(huber(recon_smo - logits_y), 1)
            loss_recon_smo += tf.expand_dims(js(recon_smo_prob, y), -1)
            loss_recon = tf.reduce_mean(loss_recon_noi + loss_recon_smo)

            loss_content_noi = huber(content_noi2smo - content_noi)
            loss_content_smo = huber(content_smo2noi - content_smo)
            loss_style_noi = huber(style_noi2smo - style_smo)
            loss_style_smo = huber(style_smo2noi - style_noi)
            loss_content = loss_content_noi + loss_content_smo
            loss_content = tf.reduce_mean(tf.reduce_mean(loss_content, -1), -1)
            loss_style = tf.reduce_mean(loss_style_noi + loss_style_smo, -1)
            loss_cross = loss_content + loss_style

            loss = tf.reduce_mean(loss_recon + loss_cross)

        gen_vars = (
            content_enc_noi.trainable_variables
            + content_enc_smo.trainable_variables
            + style_enc_noi.trainable_variables
            + style_enc_smo.trainable_variables
            + mlp_noi.trainable_variables
            + mlp_smo.trainable_variables
            + decoder_noi.trainable_variables
            + decoder_smo.trainable_variables
        )

        grads = tape.gradient(loss, gen_vars)
        gen_opt.apply_gradients(zip(grads, gen_vars))

        return loss

    @tf.function
    def gen_train_step(y, logits_y, mask_y, x, logits_x, mask_x, train=True):
        with tf.GradientTape() as tape:
            # 1. obtain the style and content codes
            content_noi = content_enc_noi(logits_x, training=True)
            content_smo = content_enc_smo(logits_y, training=True)
            style_noi = style_enc_noi(logits_x, training=True)
            style_smo = style_enc_noi(logits_y, training=True)

            # 2. obtain reconstructions
            weights_noi = mlp_noi(style_noi)
            weights_smo = mlp_smo(style_smo)
            recon_noi = decoder_noi(content_noi, weights_noi, training=True)
            recon_smo = decoder_smo(content_smo, weights_smo, training=True)
            recon_smo_prob = masked_softmax(tf.squeeze(recon_smo, -1), mask_y)

            # 3. cross domain decode
            noi2smo = decoder_smo(content_noi, weights_smo, training=True)
            smo2noi = decoder_noi(content_smo, weights_noi, training=True)

            # 4. cycle gan consistency via codes
            content_noi2smo = content_enc_smo(noi2smo, training=True)
            content_smo2noi = content_enc_noi(smo2noi, training=True)
            style_noi2smo = style_enc_smo(noi2smo, training=True)
            style_smo2noi = style_enc_noi(smo2noi, training=True)

            # 5. reconstruct for denoising/supervised loss
            logits_yhat = decoder_smo(
                content_smo2noi, weights_smo, training=True
            )
            logits_yhat = tf.squeeze(logits_yhat, -1)
            yhat = masked_softmax(logits_yhat, mask_y, -1)

            # 6. compute recon losses
            loss_recon_noi = tf.reduce_mean(huber(recon_noi - logits_x), 1)
            loss_recon_smo = tf.reduce_mean(huber(recon_smo - logits_y), 1)
            loss_recon_smo += tf.expand_dims(js(recon_smo_prob, y), -1)
            loss_content_noi = huber(content_noi2smo - content_noi)
            loss_content_smo = huber(content_smo2noi - content_smo)
            loss_style_noi = huber(style_noi2smo - style_smo)
            loss_style_smo = huber(style_smo2noi - style_noi)

            # 7. GAN loss
            loss_disc_noi = -tf.math.log(1e-4 + tf.sigmoid(disc_noi(smo2noi)))
            loss_disc_smo = -tf.math.log(1e-4 + tf.sigmoid(disc_smo(noi2smo)))

            # 8. supervised loss
            superv_loss = 1e2 * w1(yhat, y, mask_y, 0.01) + js(yhat, y)

            # 9. total loss
            loss_recon = tf.squeeze(loss_recon_noi + loss_recon_smo, -1)
            loss_content = loss_content_noi + loss_content_smo
            loss_content = tf.reduce_mean(tf.reduce_mean(loss_content, -1), -1)
            loss_style = tf.reduce_mean(loss_style_noi + loss_style_smo, -1)
            loss_disc = tf.squeeze(loss_disc_noi + loss_disc_smo, -1)
            loss = (
                loss_recon
                + 1 * loss_content
                + 1 * loss_style
                + 1 * loss_disc
                + 1 * superv_loss
            )
            loss = tf.reduce_mean(loss)

        gen_vars = (
            content_enc_noi.trainable_variables
            + content_enc_smo.trainable_variables
            + style_enc_noi.trainable_variables
            + style_enc_smo.trainable_variables
            + mlp_noi.trainable_variables
            + mlp_smo.trainable_variables
            + decoder_noi.trainable_variables
            + decoder_smo.trainable_variables
        )

        grads = tape.gradient(loss, gen_vars)

        if train:
            gen_opt.apply_gradients(zip(grads, gen_vars))

        return loss, smo2noi, recon_noi

    @tf.function
    def disc_train_step(logits_x, logits_y, train=True):

        B = logits_x.shape[0]
        with tf.GradientTape() as tape:
            content_noi = content_enc_noi(logits_x, training=False)
            content_smo = content_enc_smo(logits_y, training=False)
            noise_noi = tf.random.normal((B, style_dim))
            noise_smo = tf.random.normal((B, style_dim))
            weights_noi = mlp_noi(noise_noi, training=False)
            weights_smo = mlp_smo(noise_smo, training=False)
            noi2smo = decoder_smo(content_noi, weights_smo, training=True)
            smo2noi = decoder_noi(content_smo, weights_noi, training=True)

            loss_disc_smo2noi = -tf.math.log(
                1e-4 + 1.0 - tf.sigmoid(disc_noi(smo2noi))
            )
            loss_disc_noi2smo = -tf.math.log(
                1e-4 + 1.0 - tf.sigmoid(disc_smo(noi2smo))
            )
            loss_disc_noireal = -tf.math.log(
                1e-4 + tf.sigmoid(disc_noi(logits_x))
            )
            loss_disc_smoreal = -tf.math.log(
                1e-4 + tf.sigmoid(disc_smo(logits_y))
            )
            loss_disc_fake = tf.reduce_mean(
                loss_disc_smo2noi + loss_disc_noi2smo
            )
            loss_disc_real = tf.reduce_mean(
                loss_disc_noireal + loss_disc_smoreal
            )
            loss = loss_disc_fake + loss_disc_real

        disc_vars = disc_noi.trainable_variables + disc_smo.trainable_variables
        grads = tape.gradient(loss, disc_vars)

        if train:
            disc_opt.apply_gradients(zip(grads, disc_vars))

        return loss, loss_disc_fake, loss_disc_real

    @tf.function
    def smoother_train_step(logits_y, y, mask_y):

        B = logits_y.shape[0]
        noise = tf.random.normal((B, style_dim))
        weights_noi = mlp_noi(noise, training=False)
        content = content_enc_smo(logits_y)
        logits_fake = decoder_noi(content, weights_noi)
        with tf.GradientTape() as tape:
            logits_yhat = tf.squeeze(smoother(logits_fake, training=True), -1)
            yhat = denoiser.masked_softmax(logits_yhat, mask_y, -1)
            w_loss = js(yhat, y) + 1e2 * w1(yhat, y, mask_y, 0.01)
            loss = tf.reduce_mean(w_loss)

        grads = tape.gradient(loss, smoother.trainable_variables)
        smooth_opt.apply_gradients(zip(grads, smoother.trainable_variables))
        fake = denoiser.masked_softmax(tf.squeeze(logits_fake, -1), mask_y, -1)
        disc = tf.sigmoid(disc_noi(logits_fake))
        return loss, fake, yhat, disc

    @tf.function
    def test_step(logits_x, y, mask_y):
        logits_yhat = tf.squeeze(smoother(logits_x, training=True), -1)
        yhat = denoiser.masked_softmax(logits_yhat, mask_y, -1)
        w_loss = js(yhat, y) + 1e2 * w1(yhat, y, mask_y, 0.01)
        loss = tf.reduce_mean(w_loss)
        fake = denoiser.masked_softmax(tf.squeeze(logits_fake, -1), mask_y, -1)
        disc = tf.sigmoid(disc_noi(logits_x))

        return loss, fake, yhat, disc

    step = 0
    od_step = 0
    fake_loss_buff = []
    real_loss_buff = []
    gen_loss_buff = []
    norm_loss_buff = []
    cycle_loss_buff = []
    smo_loss_buff = []
    irm_loss_buff = []
    # idx_pmin_buff = []
    # gp_buff = []
    ood_buff = []
    test_buff = []
    epochs = cfg.epochs
    plot_every = cfg.plot_every
    batch_size = cfg.batch_size
    envs = cfg.batch_size // cfg.env_block_size
    cr = cfg.cycle_reg
    ir = cfg.identity_reg
    max_nr = cfg.max_noise_ratio

    dfake_ma = 0.0
    dreal_ma = 0.0
    train_gen = True
    train_dis = True
    # train_smo = True

    id_loss = None
    gen_opt.lr.assign(cfg.pretrain_lr)

    if not cfg.load_pretrained:
        logger.write_message("Pretraining identity....")
        id_loss_ = 0.0
        for e in range(cfg.pretrain_identity):
            j = 0
            for batch_data in batches(training_data, ood_data, batch_size):
                logits_y, logits_x, _, _, x, y, mask_y, mask_x = batch_data

                # update_gen = j % cfg.train_gen_every == 0
                id_loss = identity_train_step(
                    y, logits_y, mask_y, x, logits_x, mask_x
                )
                id_loss_ += 0.01 * (float(id_loss) - id_loss_)
            logger.write_message(f"Epoch: {e}, Identity loss {id_loss_:.2f}")
        logger.write_message("Saving pretrained")

    for e in range(epochs):

        # find learning rate
        if e < cfg.burnin:
            lr = cfg.min_lr + cfg.init_lr * e / cfg.burnin
            smooth_lr = lr
        else:
            e0 = e - cfg.burnin
            lr = cfg.init_lr * (cfg.half_power ** (e0 // cfg.half_lr_every))
            lr = max(cfg.min_lr, lr)
            smooth_lr = cfg.smooth_init_lr * (
                cfg.half_power ** (e // cfg.smooth_half_lr_every)
            )

        disc_opt.lr.assign(lr * cfg.disc_lr_multiplier)
        gen_opt.lr.assign(lr)
        smooth_opt.lr.assign(smooth_lr)

        for j, batch_data in enumerate(
            batches(training_data, ood_data, batch_size)
        ):
            logits_y, logits_x, noise, alpha, x, y, mask_y, mask_x = batch_data
            train_gen = j % cfg.train_gen_every == 0
            # train_smo = j % cfg.train_smoother_every == 0
            train_dis = j % cfg.train_disc_every == 0

            # update_gen = j % cfg.train_gen_every == 0
            gen_loss, logits_fake, recon_smo, recon_noi = gen_train_step(
                y, logits_y, mask_y, x, logits_x, mask_x, train_gen,
            )

            if cfg.use_replay:
                buffer.add(
                    logits_fake[
                        : (cfg.replay_blocks * cfg.env_block_size)
                    ].numpy(),
                    y[: (cfg.replay_blocks * cfg.env_block_size)].numpy(),
                    mask_y[: (cfg.replay_blocks * cfg.env_block_size)].numpy(),
                )

            gen_loss_buff.append(float(gen_loss))
            # norm_loss_buff.append(float(norm_loss))
            # cycle_loss_buff.append(float(cycle_loss))
            # irm_loss_buff.append(float(irm_loss))

            if step % plot_every == 0:
                fn = f"{logdir}/images/{step:05d}_translate.png"
                fake = denoiser.masked_softmax(logits_fake, mask_y)[0]
                psmo = denoiser.masked_softmax(recon_smo, mask_y)[0]
                pnoi = denoiser.masked_softmax(recon_smo, mask_y)[0]
                # title = f"D={preal:.2f}, nr={float(alpha[0]):.2f}"
                title = ""
                plot_translate_results(
                    *[u[0].numpy() for u in (y, psmo, fake, x, pnoi)],
                    title,
                    fn,
                )

            disc_loss, disc_fake, disc_real = disc_train_step(
                logits_x, logits_y, train_dis
            )
            # dfake_ma += cfg.band_lam * (disc_fake - dfake_ma)
            # dreal_ma += cfg.band_lam * (disc_real - dreal_ma)
            fake_loss_buff.append(float(disc_fake))
            real_loss_buff.append(float(disc_real))
            # gp_buff.append(float(gp))

            smo_loss, fake, yhat, disc = smoother_train_step(
                logits_y, y, mask_y
            )
            smo_loss_buff.append(float(smo_loss))

            if step % plot_every == 0:
                fn = f"{logdir}/images/{step:05d}.png"
                # f = denoiser.masked_softmax(logits_fake, mask_y)[0]
                # title = f"D={preal:.2f}, nr={float(alpha[0]):.2f}"
                title = f"D={float(disc[0]):.2f}"
                plot_results(
                    *[u.numpy() for u in (y[0], yhat[0], fake[0])], title, fn
                )

            step += 1

        for batch_data in batches(ood_data, ood_data, batch_size):
            logits_y, logits_x, noise, alpha, x, y, mask_y, mask_x = batch_data
            ood_loss, fake, yhat, disc = test_step(logits_x, y, mask_y)
            ood_buff.append(float(ood_loss))

            if od_step % plot_every == 0:
                fn = f"{logdir}/images/{od_step:05d}_test.png"
                title = f"D={float(disc[0]):.2f}"
                plot_results(
                    *[u.numpy() for u in (y[0], yhat[0], x[0])], title, fn
                )

            od_step += 1

        for batch_data in batches(test_data, test_data, batch_size):
            logits_y, logits_x, noise, alpha, x, y, mask_y, mask_x = batch_data
            test_loss, yhat, fake, disc = test_step(logits_x, y, mask_y)
            test_buff.append(float(test_loss))

        gen_loss = np.mean(gen_loss_buff)
        gen_loss_buff.clear()
        fake_loss = np.mean(fake_loss_buff)
        fake_loss_buff.clear()
        real_loss = np.mean(real_loss_buff)
        real_loss_buff.clear()
        # cycle_loss = np.mean(cycle_loss_buff)
        # cycle_loss_buff.clear()
        smo_loss = np.mean(smo_loss_buff)
        smo_loss_buff.clear()
        # irm_loss = np.mean(irm_loss_buff)
        # irm_loss_buff.clear()
        # norm_loss = np.mean(norm_loss_buff)
        # norm_loss_buff.clear()
        # idx_pmin = np.mean(idx_pmin_buff)
        # idx_pmin_buff.clear()
        # # gp = np.mean(gp_buff)
        # gp_buff.clear()
        ood = np.mean(ood_buff)
        ood_buff.clear()
        test = np.mean(test_buff)
        test_buff.clear()
        logger.write_metric({"global_step": e, "loss/gen": gen_loss})
        # logger.write_metric({"global_step": e, "loss/gp": gp})
        logger.write_metric({"global_step": e, "loss/fake": fake_loss})
        logger.write_metric({"global_step": e, "loss/real": real_loss})
        # logger.write_metric({"global_step": e, "loss/cycle": cycle_loss})
        logger.write_metric({"global_step": e, "loss/smooth": smo_loss})
        # logger.write_metric({"global_step": e, "loss/irm": irm_loss})
        # logger.write_metric({"global_step": e, "loss/norm_loss": norm_loss})
        logger.write_metric({"global_step": e, "info/lr": lr})
        # logger.write_metric({"global_step": e, "info/idx_pmin": idx_pmin})
        # logger.write_metric({"global_step": e, "info/expl": expl})
        # logger.write_metric({"global_step": e, "info/reg_wt": reg_wt})
        logger.write_metric({"global_step": e, "ood/ood": ood})
        logger.write_metric({"global_step": e, "ood/test": test})

    # step += 1


if __name__ == "__main__":
    main()
