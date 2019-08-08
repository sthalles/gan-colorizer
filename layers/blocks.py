import tensorflow as tf
from layers.conv_sn import SNConv2D
from layers.transpose_conv_sn import SNTransposeConv2D

class UpSample(tf.keras.Model):
    def __init__(self, filters, size, initializer, kernel_regularizer, apply_dropout=False):
        super(UpSample, self).__init__()
        self.apply_dropout = apply_dropout
        self.c1 = SNTransposeConv2D(filters, size, strides=2,
                                        padding='SAME',
                                        kernel_initializer=initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        use_bias=False)

        self.bn = tf.keras.layers.BatchNormalization()

        self.dropout = tf.keras.layers.SpatialDropout2D(0.5)
        self.activation = tf.keras.layers.ReLU()

    def call(self, h, sn_update, **kwargs):
        h = self.c1(h, sn_update=sn_update)
        h = self.bn(h, **kwargs)
        if self.apply_dropout:
            h = self.dropout(h)
        return self.activation(h)


class DownSample(tf.keras.Model):
    def __init__(self, filters, size, initializer, kernel_regularizer, apply_batchnorm=True):
        super(DownSample, self).__init__()
        self.c1 = SNConv2D(filters, size, strides=2, padding='SAME',
                             kernel_initializer=initializer, kernel_regularizer=kernel_regularizer, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.apply_batchnorm = apply_batchnorm
        self.activation = tf.keras.layers.ReLU()

    def call(self, h, sn_update, **kwargs):
        h = self.c1(h, sn_update=sn_update)
        if self.apply_batchnorm:
            h = self.bn(h, **kwargs)
        h = self.activation(h)
        return h