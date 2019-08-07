import tensorflow as tf
from layers.conv_sn import SNConv2D
from layers.transpose_conv_sn import SNTransposeConv2D


class DownSample(tf.keras.Model):
    def __init__(self, filters, kernel_size, activation, initializer, kernel_regularizer, apply_batchnorm=True):
        super(DownSample, self).__init__()
        self.activation = activation
        self.apply_batchnorm = apply_batchnorm

        self.conv = SNConv2D(filters, kernel_size, strides=2, padding='SAME', kernel_regularizer=kernel_regularizer,
                             kernel_initializer=initializer, use_bias=not apply_batchnorm)

        if apply_batchnorm:
            self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, sn_update, **kwargs):
        net = self.conv(x,sn_update=sn_update)

        if self.apply_batchnorm:
            net = self.bn(net, **kwargs)

        net = self.activation(net)
        return net


class UpSample(tf.keras.Model):
    def __init__(self, filters, kernel_size, activation, initializer, kernel_regularizer, apply_dropout=False):
        super(UpSample, self).__init__()

        self.activation = activation
        self.apply_dropout = apply_dropout
        self.conv = SNTransposeConv2D(filters, kernel_size, strides=2,
                                      padding='SAME',
                                      kernel_regularizer=kernel_regularizer,
                                      kernel_initializer=initializer,
                                      use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.SpatialDropout2D(0.5)

    def call(self, x, sn_update, **kwargs):
        net = self.conv(x, sn_update=sn_update)
        net = self.bn(net, **kwargs)
        if self.apply_dropout:
            net = self.dropout(net)
        return self.activation(net)
