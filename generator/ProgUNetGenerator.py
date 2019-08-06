import tensorflow as tf
from source.sn_non_local_block import SNNonLocalBlock
from source.sn_convolution_2d import Conv2DSN, Conv2DTransposeSN
from source.netblocks_v2 import UpSample, DownSample


class ProgUNetGenerator(tf.keras.Model):
  def __init__(self, n_classes, activation=tf.keras.layers.ReLU()):
    super(ProgUNetGenerator, self).__init__()
    # encoder
    self.downsample_map = {4:1, 8:2, 16:3, 32:4, 64:5, 128:6, 256:7}
    self.down_stack = [ DownSample(64, 3, apply_batchnorm=False, activation=activation),
                        DownSample(128, 3, activation=activation),
                        # self.enc_attention = SNNonLocalBlock(256)
                        DownSample(256, 3, activation=activation),
                        DownSample(512, 3, activation=activation),
                        DownSample(512, 3, activation=activation),
                        DownSample(1024, 3, activation=activation),
                        DownSample(1024, 3, activation=activation)]

    # decoder
    self.up_stack = [ UpSample(1024, 3, apply_dropout=True, activation=activation),
                      UpSample(512, 3, apply_dropout=True, activation=activation),
                      UpSample(512, 3, apply_dropout=True, activation=activation),
                      UpSample(256, 3, activation=activation),
                      UpSample(128, 3, activation=activation),
                      # self.dec_attention = SNNonLocalBlock(512)
                      UpSample(64, 3, activation=activation)]

    self.concat = tf.keras.layers.Concatenate()

    initializer = tf.keras.initializers.glorot_uniform()

    self.conv = Conv2DTransposeSN(n_classes, 4, strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh',
                                  name='final_conv')

  def call(self, x, **kwargs):
    # encoder forward
    if x.shape[2] < 256:
      n_downsamples = self.downsample_map[x.shape[2]]
    else:
      n_downsamples = 7

    # Downsampling through the model
    skips = []
    for down in self.down_stack[:n_downsamples]:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(self.up_stack[-(n_downsamples-1):], skips):
      x = up(x)
      x = self.concat([x, skip])

    return self.conv(x)
