import tensorflow as tf
from discriminator.renetblock import OptimizedBlock, Block
from source.sn_non_local_block import SNNonLocalBlock
from source.sn_convolution_2d import DenseSN


class ResnetDiscriminator(tf.keras.Model):
  def __init__(self, ch, activation=tf.keras.layers.ReLU()):
    super(ResnetDiscriminator, self).__init__()
    initializer = tf.keras.initializers.glorot_uniform()
    self.activation = activation
    self.concat = tf.keras.layers.Concatenate()
    self.block1 = OptimizedBlock(ch, ksize=3)
    self.block2 = Block(ch, ch * 2, ksize=3, downsample=True)
    self.sn_block = SNNonLocalBlock(ch * 2)
    self.block3 = Block(ch * 2, ch * 4, ksize=3, downsample=True)
    self.block4 = Block(ch * 4, ch * 8, ksize=3, downsample=True)
    self.block5 = Block(ch * 8, ch * 16, ksize=3, downsample=True)
    self.linear = DenseSN(units=1, kernel_initializer=initializer)

  def call(self, x, y, **kwargs):
    h = self.concat([x, y])
    h = self.block1(h)
    h = self.block2(h)
    h = self.sn_block(h)
    h = self.block3(h)
    h = self.block4(h)
    h = self.block5(h)
    h = self.activation(h)
    h = tf.reduce_sum(h, axis=(1, 2))
    h = self.linear(h)

    return h
