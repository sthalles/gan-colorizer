import tensorflow as tf
from source.sn_convolution_2d import Conv2DSN, Conv2DTransposeSN


# class FeatureMatching(tf.keras.Model):
#   def __init__(self, filters, kernel_size, alpha=0.2):
#     super(FeatureMatching, self).__init__()
#     initializer = tf.keras.initializers.glorot_uniform()
#     self.conv = Conv2DSN(filters=filters, kernel_size=kernel_size, strides=1, padding='valid',
#                          kernel_initializer=initializer, use_bias=True)
#     self.activation = tf.keras.layers.LeakyReLU(alpha)
#     self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')
#
#   def call(self, x):
#     net = self.conv(x)
#     net = self.activation(net)
#     return self.global_avg_pool(net)


class DownSample(tf.keras.Model):
  def __init__(self, filters, kernel_size, activation, apply_batchnorm=True):
    super(DownSample, self).__init__()
    self.activation = activation
    self.apply_batchnorm = apply_batchnorm
    initializer = tf.keras.initializers.glorot_uniform()

    self.conv = Conv2DSN(filters, kernel_size, strides=2, padding='same',
                         kernel_initializer=initializer, use_bias=not apply_batchnorm)

    if apply_batchnorm:
      self.bn = tf.keras.layers.BatchNormalization()

  def call(self, x, **kwargs):
    net = self.conv(x)

    if self.apply_batchnorm:
      net = self.bn(net, **kwargs)

    net = self.activation(net)
    return net


class UpSample(tf.keras.Model):
  def __init__(self, filters, kernel_size, activation, apply_dropout=False):
    super(UpSample, self).__init__()
    initializer = tf.keras.initializers.glorot_uniform()
    self.activation = activation
    self.apply_dropout = apply_dropout
    self.conv = Conv2DTransposeSN(filters, kernel_size, strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization()
    self.dropout = tf.keras.layers.SpatialDropout2D(0.5)

  def call(self, x, **kwargs):
    net = self.conv(x)
    net = self.bn(net, **kwargs)
    if self.apply_dropout:
      net = self.dropout(net)
    return self.activation(net)
