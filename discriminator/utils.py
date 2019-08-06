import tensorflow as tf
from source.sn_convolution_2d import Conv2DSN, DenseSN

class FeatureMatching(tf.keras.Model):
  def __init__(self, filters, kernel_size, activation):
    super(FeatureMatching, self).__init__()
    self.activation = activation
    initializer = tf.keras.initializers.glorot_uniform()
    self.conv = Conv2DSN(filters=filters, kernel_size=kernel_size, strides=1, padding='valid',
                         kernel_initializer=initializer, use_bias=True)
    self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')

  def call(self, x):
    net = self.conv(x)
    net = self.activation(net)
    return self.global_avg_pool(net)


class PatchGAN(tf.keras.Model):
  def __init__(self, filters, kernel_size, activation, apply_batchnorm=False):
    super(PatchGAN, self).__init__()
    initializer = tf.keras.initializers.glorot_uniform()
    self.apply_batchnorm = apply_batchnorm

    self.conv = Conv2DSN(filters=filters, kernel_size=kernel_size, strides=1, padding='valid',
                         kernel_initializer=initializer, use_bias=not apply_batchnorm)
    self.padding = tf.keras.layers.ZeroPadding2D()
    if apply_batchnorm:
      self.bn = tf.keras.layers.BatchNormalization()

    self.activation = activation
    self.conv2 = Conv2DSN(1, 3, strides=1, padding='valid', kernel_initializer=initializer)

  def call(self, x, **kwargs):
    net = self.padding(x)
    net = self.conv(net)

    if self.apply_batchnorm:
      net = self.bn(net, **kwargs)

    net = self.activation(net)
    net = self.padding(net)
    return self.conv2(net)