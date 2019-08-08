import tensorflow as tf
from layers.conv_sn import SNConv2D
from layers.transpose_conv_sn import SNTransposeConv2D


class DownSample(tf.keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(DownSample, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.c1 = SNConv2D(filters, size, strides=2, padding='SAME',
                           kernel_initializer=initializer, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.apply_batchnorm = apply_batchnorm
        self.activation = tf.keras.layers.LeakyReLU()

    def call(self, h, sn_update, **kwargs):
        h = self.c1(h, sn_update=sn_update)
        if self.apply_batchnorm:
            h = self.bn(h, **kwargs)
        h = self.activation(h)
        return h


class Discriminator(tf.keras.Model):

    def __init__(self, ch):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.down1 = DownSample(ch, 4, apply_batchnorm=False)  # (bs, 128, 128, 64)
        self.down2 = DownSample(ch * 2, 4)  # (bs, 64, 64, 128)
        self.down3 = DownSample(ch * 4, 4)  # (bs, 32, 32, 256)

        self.pad = tf.keras.layers.ZeroPadding2D()  # (bs, 34, 34, 256)
        self.conv = SNConv2D(ch * 8, 4, strides=1,
                             kernel_initializer=initializer,
                             use_bias=False)  # (bs, 31, 31, 512)

        self.bn = tf.keras.layers.BatchNormalization()

        self.activation = tf.keras.layers.LeakyReLU()

        self.last = SNConv2D(1, 4, strides=1,
                             kernel_initializer=initializer)  # (bs, 30, 30, 1)

    def call(self, h, sn_update, **kwargs):
        h = self.down1(h, sn_update, **kwargs)
        h = self.down2(h, sn_update, **kwargs)
        h = self.down3(h, sn_update, **kwargs)

        h = self.pad(h)
        h = self.conv(h, sn_update=sn_update)

        h = self.bn(h, **kwargs)
        h = self.activation(h)
        h = self.pad(h)

        h = self.last(h, sn_update=sn_update)
        return h
