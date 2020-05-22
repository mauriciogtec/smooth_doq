import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.initializers import TruncatedNormal


class ResBlock(tf.keras.Model):
    def __init__(self,
                 ksize,
                 filters,
                 leaky=False,
                 pooling=True,
                 noisy=False,
                 batch_norms=True,
                 virtual_batch_size=None):
        super(ResBlock, self).__init__()
        self.ksize = ksize
        self.filters = filters
        self.pooling = pooling
        self.noisy = noisy
        self.leaky = leaky
        self.batch_norms = batch_norms
        self.virtual_batch_size = virtual_batch_size
        # block 1
        if self.batch_norms:
            self.bn1 = tf.keras.layers.BatchNormalization(
                epsilon=0.1, scale=True, renorm=False,
                virtual_batch_size=self.virtual_batch_size)
        self.conv1 = tf.keras.layers.Conv1D(
            filters, 1, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.0001))
        # block 2
        if self.batch_norms:
            self.bn2 = tf.keras.layers.BatchNormalization(
                epsilon=0.1, scale=True, renorm=False,
                virtual_batch_size=self.virtual_batch_size)
            #
        self.conv2_1 = tf.keras.layers.Conv1D(
            self.filters // 4, ksize, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.0001))
        #
        k2 = max(self.ksize - 2, 1)
        self.conv2_2 = tf.keras.layers.Conv1D(
            self.filters // 4, k2, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.0001))
        #
        k2 = max(self.ksize - 4, 1)
        self.conv2_3 = tf.keras.layers.Conv1D(
            self.filters // 4, k2, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.0001))
        #
        # k2 = max(self.ksize - 6, 1)
        self.conv2_4 = tf.keras.layers.Conv1D(
            self.filters // 4, 1, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.0001))
        self.maxpool = tf.keras.layers.MaxPooling1D(
            pool_size=self.ksize - 2, strides=1, padding='same')
        #
        self.concat = tf.keras.layers.Concatenate()
        # block 3
        if self.batch_norms:
            self.bn3 = tf.keras.layers.BatchNormalization(
                epsilon=0.1, scale=True, renorm=False,
                virtual_batch_size=self.virtual_batch_size)
        self.conv3 = tf.keras.layers.Conv1D(
            self.filters, 1, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.0001))
        # addition
        self.add = tf.keras.layers.add
        self.relu4 = tf.nn.leaky_relu

    def call(self, inputs, training=False):
        x = inputs
        if self.batch_norms:
            x = self.bn1(x, training=training)
        if self.leaky:
            x = tf.nn.leaky_relu(x, alpha=0.2)
        else:
            x = tf.nn.relu(x)
        if self.noisy:
            x += tf.random.normal(tf.shape(x), stddev=0.001)
        x = self.conv1(x)
        if self.batch_norms:
            x = self.bn2(x, training=training)
        if self.leaky:
            x = tf.nn.leaky_relu(x, alpha=0.2)
        else:
            x = tf.nn.relu(x)
        if self.noisy:
            x += tf.random.normal(tf.shape(x), stddev=0.001)
        x_1 = self.conv2_1(x)
        x_2 = self.conv2_2(x)
        x_3 = self.conv2_3(x)
        x_4 = self.conv2_4(x)
        if self.pooling:
            x_4_1 = self.maxpool(x_4)
            x_4 = x_4 + x_4_1
        x = self.concat([x_1, x_2, x_3, x_4])
        if self.batch_norms:
            x = self.bn3(x, training=training)
        if self.leaky:
            x = tf.nn.leaky_relu(x, alpha=0.2)
        else:
            x = tf.nn.relu(x)
        if self.noisy:
            x += tf.random.normal(tf.shape(x), stddev=0.001)
        x = self.conv3(x)
        x = self.add([x, inputs])
        if self.leaky:
            x = tf.nn.leaky_relu(x, alpha=0.2)
        else:
            x = tf.nn.relu(x)
        if self.noisy:
            x += tf.random.normal(tf.shape(x), stddev=0.001)
        return x


class Generator(tf.keras.Model):
    def __init__(self, ksize, filters, nblocks,
                 batch_norms=False, noisy=True):
        super(Generator, self).__init__()
        self.nblocks = nblocks
        self.ksize = ksize
        self.filters = filters
        self.blocks = []

        self.conv_0 = tf.keras.layers.Conv1D(
            filters, ksize, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.0001))
        for i in range(nblocks):
            new_block = ResBlock(
                ksize, filters,
                batch_norms=batch_norms,
                pooling=False,
                leaky=False,
                noisy=True)
            self.blocks.append(new_block)
        self.gate_block = ResBlock(
            ksize, filters, batch_norms=batch_norms,
            pooling=False, noisy=False)
        self.gate_head = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same',
            activity_regularizer=tf.keras.regularizers.l2(0.001),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.0001),
            bias_initializer='zeros')
        self.noise_block = ResBlock(
            ksize, filters, batch_norms=batch_norms,
            pooling=False, noisy=noisy)
        self.noise_head = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.0001),
            bias_initializer='zeros')

    def call(self, inputs, training=False):
        noise, signal, nr = inputs
        fnoise = tf.tile(nr, (1, tf.shape(noise)[1], 1))
        x = tf.concat([noise, fnoise], -1)
        x = self.conv_0(x)
        for block in self.blocks:
            x = block(x, training=training)
        # dispersion mechanism
        x1 = x
        x1 = self.gate_block(x1, training=training)
        gate = self.gate_head(x1)
        gate = tf.clip_by_value(gate, -5.0, 5.0)
        gate += tf.random.normal(tf.shape(gate), mean=0.0, stddev=1e-3)
        x1 = signal * tf.math.exp(gate)
        x1 /= tf.reduce_sum(x1, axis=1, keepdims=True)
        x_signal = x1
        # additive noise
        x2 = x
        x2 = self.noise_block(x2, training=training)
        x2 = self.noise_head(x2)
        x2 += noise
        x2 = tf.clip_by_value(x2, 1e-8, 1e6)
        x2 /= tf.reduce_sum(x2, axis=1, keepdims=True)
        x_noise = x2
        # extra noise
        x = nr * x_noise + (1.0 - nr) * x_signal
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, ksize, filters, nblocks,
                 dropout=False, batch_norms=False):
        super(Discriminator, self).__init__()
        self.nblocks = nblocks
        self.ksize = ksize
        self.filters = filters
        self.blocks = []
        self.dropout = dropout
        for i in range(nblocks):
            block = ResBlock(
                ksize, filters,
                pooling=True, noisy=False,
                leaky=True, batch_norms=batch_norms)
            self.blocks.append(block)
        self.conv_0 = tf.keras.layers.Conv1D(
            filters, ksize, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001))
        self.global_pooling = tf.keras.layers.GlobalMaxPooling1D()
        self.dropout_layer = tf.keras.layers.Dropout(0.5)
        self.dense_final = tf.keras.layers.Dense(
            units=1,
            bias_initializer='zeros',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            # activity_regularizer=tf.keras.regularizers.l2(0.01),
            activation='linear')

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv_0(x)
        for block in self.blocks:
            x = block(x, training=training)
        if self.dropout:
            x = self.dropout_layer(x, training=training)
        x = self.global_pooling(x)
        x = self.dense_final(x)
        x = tf.clip_by_value(x, -10.0, 10.0)
        return x


class Features(tf.keras.Model):
    def __init__(self, ksize, filters, nblocks, batch_norms=False):
        super(Features, self).__init__()
        self.nblocks = nblocks
        self.ksize = ksize
        self.filters = filters
        self.blocks = []

        self.conv_0 = tf.keras.layers.Conv1D(
            filters // 2, ksize, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02))
        for i in range(nblocks):
            new_block = ResBlock(ksize, filters, batch_norms=batch_norms)
            self.blocks.append(new_block)

    def call(self, inputs):
        x = inputs
        obs = x[:, :, 0]
        obs = tf.stack((self.filters // 2) * [obs], -1)
        x = tf.concat([self.conv_0(x), obs], -1)
        for block in self.blocks:
            x = block(x)
        return x


class SignalHead(tf.keras.Model):
    def __init__(self, ksize, filters, nblocks_signal, nblocks_peaks,
                 batch_norms=True):
        super(SignalHead, self).__init__()
        self.nblocks_signal = nblocks_signal
        self.nblocks_peaks = nblocks_peaks
        self.ksize = ksize
        self.filters = filters
        self.blocks_signal = []
        self.blocks_peaks = []
        for i in range(nblocks_signal):
            self.blocks_signal.append(
                ResBlock(ksize, filters, batch_norms=batch_norms))
        for i in range(nblocks_peaks):
            self.blocks_peaks.append(
                ResBlock(ksize, filters, batch_norms=batch_norms))
        self.conv_final_signal = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            activity_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv_final_peaks = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            activity_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs, training=False):
        x, obs = inputs
        # obs = tf.expand_dims(inputs[1][:, :, 0], -1)
        # signal
        x1 = x
        for block in self.blocks_signal:
            x1 = block(x1, training=training)
        x1 = self.conv_final_signal(x1)
        x1 += 1e-6 * obs
        x1 = tf.clip_by_value(x1, 1e-9, 1e6)
        x1 /= tf.reduce_sum(x1, axis=1, keepdims=True)
        signal = x1
        # peaks
        x2 = x
        x2 += 1e-6 * obs  # add it a bit earlier
        for block in self.blocks_peaks:
            x2 = block(x2, training=training)
        peaks = self.conv_final_peaks(x2)
        peaks = tf.clip_by_value(peaks, -10.0, 10.0)
        return signal, peaks


class DeconvHead(tf.keras.Model):
    def __init__(self, ksize, filters, nblocks, batch_norms=True):
        super(DeconvHead, self).__init__()
        self.nblocks = nblocks
        self.ksize = ksize
        self.filters = filters
        self.blocks = []
        for i in range(nblocks):
            self.blocks.append(
                ResBlock(ksize, filters, batch_norms=batch_norms))
        self.conv_final_signal = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            activity_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs, training=False):
        x, _ = inputs
        obs = tf.expand_dims(inputs[1][:, :, 0], -1)
        # signal
        x1 = x
        for block in self.blocks:
            x1 = block(x1, training=training)
        x1 = self.conv_final_signal(x1)
        x1 += 1e-6 * obs
        x1 = tf.clip_by_value(x1, 1e-9, 1e6)  # capped relu
        x1 /= tf.reduce_sum(x1, axis=1, keepdims=True)
        signal = x1
        return signal


class DeconvHead2(tf.keras.Model):
    def __init__(self, ksize, filters,
                 nblocks_signal, nblocks_weights, batch_norms=True):
        super(DeconvHead2, self).__init__()
        self.nblocks_signal = nblocks_signal
        self.nblocks_weights = nblocks_weights
        self.ksize = ksize
        self.filters = filters
        self.blocks_signal = []
        self.blocks_weights = []
        for i in range(nblocks_signal):
            self.blocks_signal.append(
                ResBlock(ksize, filters, batch_norms=batch_norms))
        for i in range(nblocks_weights):
            self.blocks_weights.append(
                ResBlock(ksize, filters, batch_norms=batch_norms))
        self.conv_final_signal = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            activity_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense_final_weights = tf.keras.layers.Dense(
            units=1,
            bias_initializer='zeros',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            activity_regularizer=tf.keras.regularizers.l2(0.01),
            activation='linear')

    def call(self, inputs, training=False):
        x, _ = inputs
        # signal
        x1 = x
        for block in self.blocks_signal:
            x1 = block(x1, training=training)
        x1 = self.conv_final_signal(x1)
        x1 = tf.clip_by_value(x1, 1e-9, 1e6)  # capped relu
        x1 /= tf.reduce_sum(x1, axis=1, keepdims=True)
        signal = x1
        # signal
        x2 = x
        for block in self.blocks_weights:
            x2 = block(x2, training=training)
        x2 = tf.reduce_max(x2, 1)
        x2 = self.dense_final_weights(x2)
        x2 = tf.clip_by_value(x2, -10.0, 10.0)
        weights = x2
        return signal, weights
