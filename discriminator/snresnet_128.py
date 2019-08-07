import tensorflow as tf
from layers.dense_sn import SNDense
from discriminator.resblocks import OptimizedBlock, Block
from layers.sn_non_local_block import SNNonLocalBlock

class SNResNetProjectionDiscriminator(tf.keras.Model):
    def __init__(self, ch=64, activation=tf.keras.layers.ReLU()):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.activation = activation
        initializer = tf.keras.initializers.glorot_uniform()

        self.block1 = OptimizedBlock(ch)
        self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
        self.self_atten = SNNonLocalBlock(ch * 2)
        self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
        self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
        self.block5 = Block(ch * 8, ch * 8, activation=activation, downsample=True)
        self.block6 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
        self.l6 = SNDense(units=1, kernel_initializer=initializer)

    def __call__(self, x, y=None, sn_update=None):
        assert sn_update is not None, "Define the 'sn_update' parameter"
        h = x
        h = self.block1(h, sn_update=sn_update)
        h = self.block2(h, sn_update=sn_update)
        h = self.self_atten(h, sn_update=sn_update)
        h = self.block3(h, sn_update=sn_update)
        h = self.block4(h, sn_update=sn_update)
        h = self.block5(h, sn_update=sn_update)
        h = self.block6(h, sn_update=sn_update)
        h = self.activation(h)
        h = tf.reduce_sum(h, axis=(1, 2))  # Global pooling
        output = self.l6(h, sn_update=sn_update)
        return output