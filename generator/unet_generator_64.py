import tensorflow as tf
from source.netblocks import downsample, upsample
from source.sn_non_local_block import SNNonLocalBlock
from source.sn_convolution_2d import Conv2DSN, Conv2DTransposeSN

def Generator(n_classes, in_depth, activation):
  down_stack = [
    downsample(128, 3, apply_batchnorm=False, activation=activation),  # (bs, 64, 64, 128)
    downsample(256, 3, activation=activation),  # (bs, 32, 32, 128)
    # SNNonLocalBlock(256),
    downsample(256, 3, activation=activation),  # (bs, 16, 16, 256)
    downsample(512, 3, activation=activation),  # (bs, 8, 8, 512)
    downsample(512, 3, activation=activation),  # (bs, 4, 4, 512)
    downsample(1024, 3, activation=activation),  # (bs, 2, 2, 1024)
    downsample(1024, 3, activation=activation)  # (bs, 1, 1, 1024)
  ]
  upAttention = SNNonLocalBlock(256)

  up_stack = [
    upsample(1024, 3, apply_dropout=True, activation=activation),  # (bs, 2, 2, 2048)
    upsample(512, 3, apply_dropout=True, activation=activation),  # (bs, 4, 4, 1024)
    upsample(512, 3, apply_dropout=True, activation=activation),  # (bs, 8, 8, 1024)
    upsample(256, 3, activation=activation),  # (bs, 16, 16, 512)
    upsample(256, 3, activation=activation),  # (bs, 32, 32, 256)
    # SNNonLocalBlock(512),
    upsample(128, 3, activation=activation),  # (bs, 64, 64, 512)
  ]
  downAttention = SNNonLocalBlock(256)

  initializer = tf.keras.initializers.glorot_uniform()
  last = Conv2DTransposeSN(n_classes, 4, strides=2,
                           padding='same',
                           kernel_initializer=initializer,
                           activation='tanh')  # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None, None, in_depth])
  x = inputs

  # Downsampling through the model
  skips = []
  c = 0
  for down in down_stack:
    x = down(x)

    if c == 1:
      x = upAttention(x)

    skips.append(x)
    c += 1

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  c = 0
  for up, skip in zip(up_stack, skips):
    x = up(x)

    if c == 4:
      x = downAttention(x)

    x = concat([x, skip])
    c += 1

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
