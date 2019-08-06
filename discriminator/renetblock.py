import math
import tensorflow as tf
from source.sn_convolution_2d import Conv2DSN


class Block(tf.keras.Model):
  def __init__(self, in_channels, out_channels, hidden_channels=None,
               ksize=3, pad="same", downsample=False, activation=tf.keras.layers.ReLU()):
    super(Block, self).__init__()
    initializer = tf.keras.initializers.glorot_uniform()
    self.activation = activation
    self.downsample = downsample
    self.learnable_sc = (in_channels != out_channels) or downsample
    hidden_channels = in_channels if hidden_channels is None else hidden_channels

    self.c1 = Conv2DSN(hidden_channels, kernel_size=ksize, padding=pad, kernel_initializer=initializer)
    self.c2 = Conv2DSN(out_channels, kernel_size=ksize, padding=pad, kernel_initializer=initializer)
    self._downsample = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="same")

    if self.learnable_sc:
      self.c_sc = Conv2DSN(out_channels, kernel_size=1, padding="valid", kernel_initializer=initializer)

  def residual(self, x):
    h = x
    h = self.activation(h)
    h = self.c1(h)
    h = self.activation(h)
    h = self.c2(h)
    if self.downsample:
      h = self._downsample(h)
    return h

  def shortcut(self, x):
    if self.learnable_sc:
      x = self.c_sc(x)
      if self.downsample:
        return self._downsample(x)
      else:
        return x
    else:
      return x

  def __call__(self, x):
    return self.residual(x) + self.shortcut(x)


class OptimizedBlock(tf.keras.Model):
  def __init__(self, out_channels, ksize=3, pad="same", activation=tf.keras.layers.ReLU()):
    super(OptimizedBlock, self).__init__()
    initializer = tf.keras.initializers.glorot_uniform()
    self.activation = activation

    self.c1 = Conv2DSN(out_channels, kernel_size=ksize, padding=pad, kernel_initializer=initializer)
    self.c2 = Conv2DSN(out_channels, kernel_size=ksize, padding=pad, kernel_initializer=initializer)
    self.c_sc = Conv2DSN(out_channels, kernel_size=1, padding="valid", kernel_initializer=initializer)
    self._downsample = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="valid")

  def residual(self, x):
    h = x
    h = self.c1(h)
    h = self.activation(h)
    h = self.c2(h)
    h = self._downsample(h)
    return h

  def shortcut(self, x):
    return self.c_sc(self._downsample(x))

  def __call__(self, x):
    return self.residual(x) + self.shortcut(x)
