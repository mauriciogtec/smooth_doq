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
from tensorflow_addons.optimizers import AdamW

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
        ix_b = np.random.randint(
            0, self._next_idx // self._block, size=blocks
        )
        ix_b = (ix_b * self._block) % self._N
        ix = []
        for i in ix_b:
            for j in range(self._block):
                ix.append(i + j)
        return self._storage1[ix], self._storage2[ix], self._storage3[ix]

    def add(self, block1: np.ndarray, block2: np.ndarray, block3: np.ndarray) -> None:
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
    logits_y = tf.clip_by_value(logits_y, -C, C)
    logits_x = tf.math.log(ls + x)
    logits_x -= tf.reduce_mean(logits_x, -1, keepdims=True)
    logits_x = tf.clip_by_value(logits_x, -C, C)

    noise = tf.random.uniform(logits_y.shape)
    envs = logits_x.shape[0] // ebsize
    alpha = tf.repeat(
        tf.random.uniform(shape=(envs,), minval=0.00, maxval=1.0),
        ebsize
    )

    return logits_y, logits_x, noise, alpha, x, y, mask_y, mask_x


def kld(y, yhat, eps=1e-6):
    out = -y * (tf.math.log(yhat + eps) - tf.math.log(y + eps))
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


@hydra.main(config_name="6_free_gan_F.yml")
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
    ood_data = [
        ood_data[i] for i in range(int(0.9 * len(ood_data)))
    ]

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

    nblocks = 2
    # kernel_size = [7] + [7] * nblocks
    kernel_size = [21, 3, 7]  # + [7] * nblocks
    filters = 128
    dilation_rate = [1, 8, 1] #+ [1, 2, 4, 8, 1, 1, 1, 1]
    strides = [2, 2, 2]

    generator = gan.ParamGAN(
        nblocks=nblocks,
        noise_pre_kernel_size=5,
        kernel_size=kernel_size,
        filters=filters,
        dilation_rate=dilation_rate,
        normalization=cfg.gen_norm_layers,
        dropout=cfg.gen_dropout
    )

    discriminator = gan.Discriminator(
        nblocks=nblocks,
        kernel_size=kernel_size,
        filters=filters,
        dilation_rate=dilation_rate,
        strides=strides,
    )

    discriminator2 = gan.Discriminator(
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

    if cfg.use_replay:
        buffer = Replay(cfg.replay_capacity, 400, cfg.env_block_size)

    # lr assigned inside the loop
    gen_opt = AdamW(1e-4, 1.0, beta_1=0.5, clipnorm=cfg.clipnorm)
    disc_opt = AdamW(1e-4, 1.0, beta_1=0.01, clipnorm=cfg.clipnorm)
    smooth_opt = AdamW(1e-4, 1.0, beta_1=0.01, clipnorm=cfg.clipnorm)


    @tf.function
    def gen_train_step(
        noise,
        alpha,
        y,
        logits_y,
        mask_y,
        x,
        logits_x,
        mask_x,
        reg_wt=1.0,
        cr=1.0,
        expl=0.1,
        max_nr=1.0,
        nwt=1.0,
        train=True
    ):
        # B = logits_y.shape[0]
        smo_loss = 0.0
        norm_loss = 0.0
        C = 20

        # for cycle loss
        logits_xhat = smoother(logits_x, training=False)

        with tf.GradientTape() as tape:
            # nmult, nadd = generator(logits_y, noise, training=True)
            nadd = generator(logits_y, noise, alpha, training=True)
            noise = mask_y * tf.tanh(nadd / C) * 2 * C
            logits_fake = logits_y + noise
            fake = denoiser.masked_softmax(logits_fake, mask_y)
   
            # intervent on noise to signal ratio
            # scale = 1.0
            norm_signal = tf.reduce_sum((logits_y * mask_y)**2, -1) + 1e-6
            norm_noise = tf.reduce_sum(noise**2, -1) + 1e-6
            scale = norm_noise / norm_signal
            norm_loss = 0.1 * tf.reduce_mean(huber(scale - alpha))

            # for cycle loss
            nadd_x = generator(logits_xhat, noise, alpha, training=True)

            # scale = max_nr * (norm_signal / norm_noise) * alpha
            # scale = tf.stop_gradient(tf.expand_dims(scale, -1))
            # fake = logits_y + scale * g_noise
            # fake = logits_y + g_noise

            w = tf.constant(1.0)
            with tf.GradientTape() as dummy_tape:
                dummy_tape.watch(w)
                disc = w * discriminator(logits_fake, training=False)
                disc2 = 0.01 * w * discriminator2(noise, training=False)
                logits_yhat = w * smoother(logits_fake, training=True)
                yhat = denoiser.masked_softmax(logits_yhat, mask_y)

                gen_loss = - tf.math.log(tf.sigmoid(disc + disc2) + 1e-6)
                # norm_loss = 0.001 * norm_noise
                # norm_loss -= 0.01 * tf.math.log(norm_noise + 1e-10)
                # d = tf.math.abs(norm_noise - 1.0)
                # norm_loss = tf.where(d < 1.0, 0.5 * d**2, d - 0.5)
 
                # for cycle loss
                noise_x = mask_x * tf.tanh(w * nadd_x / C) * 2 * C
                xrecon = denoiser.masked_softmax(logits_xhat + noise_x, mask_x)
                
                cycle_loss_x = 1e2 * w1(xrecon, x, mask_x, 0.01)
                cycle_loss_y = 1e2 * w1(yhat, y, mask_y, 0.01) + js(yhat, y)
                cycle_loss = cycle_loss_y +  cycle_loss_x

                w_loss = gen_loss + cr * cycle_loss

            # smo_loss = tf.reduce_mean(smo_loss)
            
            loss = tf.reduce_mean(w_loss)
            # norm_loss = tf.reduce_mean(norm_loss)
            cycle_loss = tf.reduce_mean(cycle_loss)
            gen_loss = tf.reduce_mean(gen_loss)

            # irm loss
            g = dummy_tape.jacobian(w_loss, w)
            # g = tf.clip_by_value(g, -3, 3)  # compress
            irm_loss = 0.0
            for g_e in tf.split(g, envs):
                g1 = tf.reduce_mean(g_e[::2])
                g2 = tf.reduce_mean(g_e[1::2])
                irm_loss += g1 * g2  / envs # + 0.1 * (g1 + g2) ** 2

            loss += reg_wt * irm_loss + nwt * norm_loss

        grads = tape.gradient(loss, generator.trainable_variables)

        if train:
            gen_opt.apply_gradients(zip(grads, generator.trainable_variables))

        return gen_loss, cycle_loss, fake, logits_fake, disc, yhat, norm_loss, irm_loss

    @tf.function
    def disc_train_step(logits_x, logits_fake, logits_y, reg_wt, train=True):

        logits_xhat = smoother(logits_x, training=False)
        # xhat_diff = logits_x - logits_xhat
        # fake_diff = logits_fake - logits_y    
        with tf.GradientTape() as tape:
            w = tf.constant(1.0)
            with tf.GradientTape() as dummy_tape:
                dummy_tape.watch(w)
                disc_fake = w * discriminator(logits_fake, training=True)
                disc_real = w * discriminator(logits_x, training=True)
                disc_fake2 = 0.01 * w * discriminator2(logits_fake - logits_y, training=True)
                disc_real2 = 0.01 * w * discriminator2(logits_x - logits_xhat, training=True)
                disc_fake = tf.math.log(1.0 - tf.sigmoid(disc_fake + disc_fake2) + 1e-2)
                disc_real = tf.math.log(0.99 * tf.sigmoid(disc_real + disc_real2) + 1e-4)
                w_loss = - disc_real - disc_fake
            disc_fake = tf.reduce_mean(disc_fake)
            disc_real = tf.reduce_mean(disc_real)
            loss = - disc_real - disc_fake
            g = dummy_tape.jacobian(w_loss, w)
            # g = tf.clip_by_value(g, -3, 3)  # compress
            irm_loss = 0.0
            for g_e in tf.split(g, envs):
                g1 = tf.reduce_mean(g_e[::2])
                g2 = tf.reduce_mean(g_e[1::2])
                irm_loss += g1 * g2 / envs  # + 0.1 * (g1 + g2) ** 2
            loss += reg_wt * irm_loss

        grads = tape.gradient(loss, discriminator.trainable_variables)

        if train:
            disc_opt.apply_gradients(
                zip(grads, discriminator.trainable_variables)
            )

        return loss, disc_fake, disc_real

    @tf.function
    def smoother_train_step(
        logits_fake, y, mask_y, logits_x, x,
        mask_x, reg_wt, envs=1, C=20, cr=1.0, bs=32
    ):
        irm_loss = 0.0

        with tf.GradientTape() as tape:
            w = tf.constant(1.0)
            logits_yhat = smoother(logits_fake, training=True)
            yhat = denoiser.masked_softmax(logits_yhat, mask_y, -1)
            # for cycle loss
            # logits_xhat = smoother(logits_x, training=True)
            # nadd_x = generator(logits_xhat, noise, training=False)
            # noise_x = mask_x * tf.tanh(nadd_x / C) * 2 * C
            # xrecon = denoiser.masked_softmax(logits_xhat + noise_x, mask_x)
            
            # cycle_loss_x = tf.reduce_mean(w1(xrecon, x, mask_x, 0.01))
            # cycle_loss_y = tf.reduce_mean(w1(yhat, y, mask_y, 0.01) + js(yhat, y))
            # cycle_loss = cycle_loss_y  # +  cycle_loss_x

            # with tf.GradientTape() as dummy_tape:
            #     dummy_tape.watch(w)
            w_loss = js(w * yhat, y) + 1e2 * w1(w * yhat, y, mask_y, 0.01)
            loss = tf.reduce_mean(w_loss)  # + cr * cycle_loss
            # irm loss
            # g = dummy_tape.jacobian(w_loss, w)
            # # g = tf.tanh(g)
            # irm_loss = 0.0
            # for g_e in tf.split(g, envs):
            #     g1 = tf.reduce_mean(g_e[::2])
            #     g2 = tf.reduce_mean(g_e[1::2])
            #     irm_loss += g1 * g2  # + 0.1 * (g1 + g2) ** 2

            # supervised smoother loss
            # loss += reg_wt * irm_loss
        
        grads = tape.gradient(loss, smoother.trainable_variables)
        smooth_opt.apply_gradients(zip(grads, smoother.trainable_variables))

        return loss, irm_loss, yhat

    @tf.function
    def batched_smoother_train_step(
        logits_y, y, mask_y, logits_x, x,
        mask_x, reg_wt, envs=1, C=20, cr=1.0, bs=32
    ):
        irm_loss = 0.0
        
        noise = tf.random.uniform(logits_y.shape)
        nadd = generator(logits_y, noise, alpha, training=False)
        noise_fake = mask_y * tf.tanh(nadd / C) * 2 * C
        logits_fake = logits_y + noise_fake

        num_updates = logits_y.shape[0] // bs

        losses = 0.0

        for update in range(num_updates):
            ix = [bs * update + j for j in range(bs)]
            smooth_loss, irm_loss, yhat = smoother_train_step(
                tf.gather(logits_fake, ix), 
                tf.gather(y, ix),
                tf.gather(mask_y, ix),
                tf.gather(logits_x, ix),
                tf.gather(x, ix),
                tf.gather(mask_x, ix),
                reg_wt, train_smo, cr=cfg.cycle_reg
            )

            losses += float(smooth_loss) / num_updates

        return losses

    @tf.function
    def test_step(logits, y, mask):
        yhat = smoother(logits, training=False)
        yhat = denoiser.masked_softmax(yhat, mask, -1)
        loss = tf.reduce_mean(js(yhat, y))
        return loss, yhat

    step = 0
    od_step = 0
    fake_loss_buff = []
    real_loss_buff = []
    gen_loss_buff = []
    norm_loss_buff = []
    cycle_loss_buff = []
    smooth_loss_buff = []
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
    train_disc = True
    train_smo = True

    for e in range(epochs):
        rate = np.exp(-e * np.log(2) / cfg.half_lr)
        # lr = cfg.min_lr + (cfg.init_lr - cfg.min_lr) * rate
        lr = cfg.init_lr *  (cfg.half_power ** (e // cfg.half_lr_every))
        smooth_lr = cfg.smooth_init_lr *  (cfg.half_power ** (e // cfg.smooth_half_lr_every))
        # e0 = max(e - cfg.reg_warmup, 0)
        reg_wt = cfg.max_reg if e >= cfg.reg_warmup else 0.0 # * (1.0 - np.exp(-e0 * np.log(2) / cfg.half_reg))
        disc_opt.lr.assign(lr * cfg.disc_lr_multiplier)
        gen_opt.lr.assign(lr)
        smooth_opt.lr.assign(smooth_lr)
        rate = np.exp(-e * np.log(2) / cfg.half_expl)
        expl = cfg.min_expl + (cfg.init_expl - cfg.min_expl) * rate

        for j, batch_data in enumerate(
            batches(training_data, ood_data, batch_size)
        ):
            logits_y, logits_x, noise, alpha, x, y, mask_y, mask_x = batch_data
            train_gen = j % cfg.train_gen_every == 0
            train_smo = j % cfg.train_smoother_every == 0
            train_dis = j % cfg.train_disc_every == 0

            # update_gen = j % cfg.train_gen_every == 0
            gen_loss, cycle_loss, fake, logits_fake, gen_disc, yhat, norm_loss, irm_loss = gen_train_step(
                noise, alpha, y, logits_y, mask_y, x, logits_x, mask_x,
                reg_wt, cr, expl, max_nr, cfg.norm_reg, train_gen
            )

            if cfg.use_replay:
                buffer.add(
                    logits_fake[:(cfg.replay_blocks * cfg.env_block_size)].numpy(),
                    y[:(cfg.replay_blocks * cfg.env_block_size)].numpy(),
                    mask_y[:(cfg.replay_blocks * cfg.env_block_size)].numpy()
                )

            gen_loss_buff.append(float(gen_loss))
            norm_loss_buff.append(float(norm_loss))
            cycle_loss_buff.append(float(cycle_loss))
            irm_loss_buff.append(float(irm_loss))

            for _ in range(cfg.smooth_train_steps):
                smooth_loss = batched_smoother_train_step(
                    logits_y, y, mask_y, logits_x, x, mask_x,
                    reg_wt, train_smo, cr=cfg.cycle_reg, bs=cfg.smooth_batch_size
                )
                smooth_loss_buff.append(float(smooth_loss))

            # num_steps = cfg.batch_size // cfg.smooth_batch_size
            # for k in range(num_steps):
            #     ix = [
            #         cfg.smooth_batch_size * k + j
            #         for j in range(cfg.smooth_batch_size)
            #     ]
            #     smooth_loss, irm_loss, yhat = smoother_train_step(
            #         tf.gather(logits_fake, ix), 
            #         tf.gather(y, ix),
            #         tf.gather(mask_y, ix),
            #         tf.gather(logits_x, ix),
            #         tf.gather(x, ix),
            #         tf.gather(mask_x, ix),
            #         reg_wt, train_smo, cr=cfg.cycle_reg
            #     )
            #     smooth_loss_buff.append(float(smooth_loss))


            if cfg.use_replay:
                blocks = cfg.batch_size // (2 * cfg.env_block_size)
                logits_fake_replay, y_replay, mask_replay = buffer.sample(blocks)
                logits_fake_replay = tf.constant(logits_fake_replay)
                y_replay = tf.constant(y_replay)
                logits_fake_mix = tf.concat(
                    [logits_fake_replay, logits_fake[:(cfg.batch_size // 2)]], 0
                )
                y_mix = tf.concat([y_replay, y[:(cfg.batch_size // 2)]], 0)
                mask_mix = tf.concat(
                    [mask_replay, mask_y[:(cfg.batch_size // 2)]], 0
                )
            else:
                logits_fake_mix = logits_fake
                y_mix = y
                mask_mix = mask_y
            
            
            disc_loss, disc_fake, disc_real = disc_train_step(
                logits_x, logits_fake_mix, reg_wt, train_disc
            )
            # dfake_ma += cfg.band_lam * (disc_fake - dfake_ma)
            # dreal_ma += cfg.band_lam * (disc_real - dreal_ma)
            fake_loss_buff.append(float(disc_fake))
            real_loss_buff.append(float(disc_real))
            # gp_buff.append(float(gp))

            # # whether train disc/gen or not in next rounds to guarantee eq
            # if train_gen and (dfake_ma > dreal_ma or dfake_ma > cfg.band_lim):
            #     train_gen = False
            #     train_disc = True
            #     msg = f"gen off dreal={dreal_ma:.2f}, dfake={dfake_ma:.2f}"
            #     logger.write_message(msg)
            # elif not train_gen and dreal_ma > 0 and dfake_ma < cfg.band_tgt:
            #     train_gen = True
            #     train_disc = True
            #     msg = f"gen on dreal={dreal_ma:.2f}, dfake={dfake_ma:.2f}"
            #     logger.write_message(msg)
            # elif train_disc and dreal_ma > 0 and dfake_ma < -cfg.band_lim:
            #     train_gen = True
            #     train_disc = False
            #     msg = f"disc off dreal={dreal_ma:.2f}, dfake={dfake_ma:.2f}"
            #     logger.write_message(msg)
            # elif (
            #     not train_disc
            #     and dfake_ma < dreal_ma
            #     and dfake_ma > -cfg.band_tgt
            # ):
            #     train_gen = True
            #     train_disc = True
            #     msg = f"disc on dreal={dreal_ma:.2f}, dfake={dfake_ma:.2f}"
            #     logger.write_message(msg)

            if step % plot_every == 0:
                fn = f"{logdir}/images/{step:05d}.png"
                preal = float(tf.sigmoid(gen_disc[0]))
                # f = denoiser.masked_softmax(logits_fake, mask_y)[0]
                title = f"D={preal:.2f}, nr={float(alpha[0]):.2f}"
                plot_results(
                    *[u.numpy() for u in (y[0], yhat[0], fake[0])], title, fn
                )

            step += 1

        for batch_data in batches(ood_data, ood_data, batch_size):
            logits_y, logits_x, noise, alpha, x, y, mask, mask_x = batch_data
            ood_loss, yhat = test_step(logits_x, y, mask_y)
            ood_buff.append(float(ood_loss))

            if od_step % plot_every == 0:
                fn = f"{logdir}/images/{od_step:05d}_test.png"
                preal = float(tf.sigmoid(gen_disc[0]))
                title = f"Test D={preal:.2f}"
                plot_results(
                    *[u.numpy() for u in (y[0], yhat[0], x[0])], title, fn
                )

            od_step += 1

        for batch_data in batches(test_data, test_data, batch_size):
            logits_y, logits_x, noise, alpha, x, y, mask_y, mask_x = batch_data
            test_loss, yhat = test_step(logits_x, y, mask_y)
            test_buff.append(float(test_loss))

        gen_loss = np.mean(gen_loss_buff)
        gen_loss_buff.clear()
        fake_loss = np.mean(fake_loss_buff)
        fake_loss_buff.clear()
        real_loss = np.mean(real_loss_buff)
        real_loss_buff.clear()
        cycle_loss = np.mean(cycle_loss_buff)
        cycle_loss_buff.clear()
        smooth_loss = np.mean(smooth_loss_buff)
        smooth_loss_buff.clear()
        irm_loss = np.mean(irm_loss_buff)
        irm_loss_buff.clear()
        norm_loss = np.mean(norm_loss_buff)
        norm_loss_buff.clear()
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
        logger.write_metric({"global_step": e, "loss/cycle": cycle_loss})
        logger.write_metric({"global_step": e, "loss/smooth": smooth_loss})
        logger.write_metric({"global_step": e, "loss/irm": irm_loss})
        logger.write_metric({"global_step": e, "loss/norm_loss": norm_loss})
        logger.write_metric({"global_step": e, "info/lr": lr})
        # logger.write_metric({"global_step": e, "info/idx_pmin": idx_pmin})
        logger.write_metric({"global_step": e, "info/expl": expl})
        logger.write_metric({"global_step": e, "info/reg_wt": reg_wt})
        logger.write_metric({"global_step": e, "ood/ood": ood})
        logger.write_metric({"global_step": e, "ood/test": test})

    # step += 1


if __name__ == "__main__":
    main()
