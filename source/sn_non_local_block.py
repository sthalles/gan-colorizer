import tensorflow as tf
from tensorflow.python.keras.layers import MaxPool2D
from source.sn_convolution_2d import Conv2DSN


class SNNonLocalBlock(tf.keras.layers.Layer):
  def __init__(self, n_channels, subsampling=True):
    super(SNNonLocalBlock, self).__init__()
    self.subsampling = subsampling
    self.theta = Conv2DSN(n_channels // 8, 1,
                          strides=1, padding='same', name="theta",
                          activation=None, use_bias=False)

    self.phi = Conv2DSN(n_channels // 8, 1,
                        strides=1, padding='same', name="phi",
                        activation=None, use_bias=False)

    self.max_pool = MaxPool2D(pool_size=2, strides=2)

    self.g = Conv2DSN(n_channels // 2, 1,
                      strides=1, padding='same', name="g",
                      activation=None, use_bias=False)

    self.sigma = self.add_weight(shape=(),
                                 name="sigma",
                                 initializer='zeros',
                                 trainable=True)

    self.conv = Conv2DSN(filters=n_channels, kernel_size=1, padding='valid', strides=1, activation=None)

  def call(self, x):
    # get the input shape
    batch_size, h, w, num_channels = x.shape

    location_num = h * w
    downsampled_num = location_num

    # theta path
    theta = self.theta(x)
    theta = tf.reshape(theta, shape=[batch_size, location_num, num_channels // 8])

    # phi path
    phi = self.phi(x)
    if self.subsampling:
      downsampled_num //= 4
      phi = self.max_pool(phi)

    phi = tf.reshape(phi, shape=[batch_size, downsampled_num, num_channels // 8])

    attn_map = tf.matmul(theta, phi, transpose_b=True)
    # The softmax operation is performed on each row
    attn_map = tf.nn.softmax(attn_map, axis=-1)

    # g path
    g = self.g(x)

    if self.subsampling:
      g = self.max_pool(g)

    g = tf.reshape(g, shape=[batch_size, downsampled_num, num_channels // 2])
    attn_g = tf.matmul(attn_map, g)
    attn_g = tf.reshape(attn_g, shape=[batch_size, h, w, num_channels // 2])

    attn_g = self.conv(attn_g)
    return x + self.sigma * attn_g
