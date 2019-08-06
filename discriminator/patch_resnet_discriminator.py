import tensorflow as tf
from discriminator.renetblock import OptimizedBlock, Block
from discriminator.utils import PatchGAN, FeatureMatching


class PatchResnetDiscriminator(tf.keras.Model):
  def __init__(self, ch, activation=tf.keras.layers.ReLU()):
    super(PatchResnetDiscriminator, self).__init__()
    initializer = tf.keras.initializers.glorot_uniform()
    self.activation = activation
    self.concat = tf.keras.layers.Concatenate()
    self.block1 = OptimizedBlock(ch, ksize=3)
    self.block2 = Block(ch, ch * 2, ksize=3, downsample=True)
    self.block3 = Block(ch * 2, ch * 4, ksize=3, downsample=True)
    self.block4 = Block(ch * 4, ch * 8, ksize=3, downsample=False)
    # self.block5 = Block(ch * 8, ch * 8, ksize=3, downsample=False)

    # self.linear = DenseSN(units=1, kernel_initializer=initializer)
    # self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')
    self.patch_gan = PatchGAN(filters=ch * 8, kernel_size=4,
                              activation=activation, apply_batchnorm=False)

  def call(self, x, y, **kwargs):
    h = self.concat([x, y])
    h = self.block1(h)
    h = self.block2(h)
    h = self.block3(h)
    h = self.block4(h)
    # h = self.block5(h)
    features = self.patch_gan(h)
    return features
