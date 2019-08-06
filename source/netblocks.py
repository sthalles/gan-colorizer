import tensorflow as tf
from source.sn_convolution_2d import Conv2DSN, Conv2DTransposeSN

def downsample(filters, kernel_size, apply_batchnorm=True, activation="relu"):
  initializer = tf.keras.initializers.glorot_uniform()

  result = tf.keras.Sequential()
  result.add(Conv2DSN(filters, kernel_size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  if activation == 'lrelu':
    result.add(tf.keras.layers.LeakyReLU(0.1))
  elif activation == "relu":
    result.add(tf.keras.layers.ReLU())

  return result


def upsample(filters, size, apply_dropout=False, activation="relu"):
  initializer = tf.keras.initializers.glorot_uniform()

  result = tf.keras.Sequential()
  result.add(Conv2DTransposeSN(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.SpatialDropout2D(0.5))

  if activation == 'lrelu':
    result.add(tf.keras.layers.LeakyReLU(0.1))
  elif activation == "relu":
    result.add(tf.keras.layers.ReLU())

  return result
