import tensorflow as tf
from layers.conv_sn import SNConv2D
from discriminator.resblocks import OptimizedBlock, Block
from layers.sn_non_local_block import SNNonLocalBlock

class SNResNetPatchGanDiscriminator(tf.keras.Model):
    def __init__(self, ch=64, activation=tf.keras.layers.ReLU()):
        super(SNResNetPatchGanDiscriminator, self).__init__()
        self.activation = activation
        initializer = tf.keras.initializers.glorot_uniform()

        self.block1 = OptimizedBlock(ch * 2, ksize=4)
        self.block2 = Block(ch * 2, ch * 4, ksize=4, activation=activation, downsample=True)
        self.block3 = Block(ch * 4, ch * 8, ksize=4, activation=activation, downsample=True)
        self.self_atten = SNNonLocalBlock(ch * 8)

        self.bn = tf.keras.layers.BatchNormalization()
        self.c1 = SNConv2D(filters=ch * 8, kernel_size=4, strides=1, padding="VALID",
                           kernel_initializer=initializer, use_bias=False)

        self.pad = tf.keras.layers.ZeroPadding2D()
        self.c2 = SNConv2D(1, 4, strides=1, padding="VALID", kernel_initializer=initializer)


    def __call__(self, x, y=None, sn_update=None):
        assert sn_update is not None, "Define the 'sn_update' parameter"
        h = x
        h = self.block1(h, sn_update=sn_update) # (bs, 128, 128, DEPTH)
        h = self.block2(h, sn_update=sn_update) # (bs, 64, 64, DEPTH)
        h = self.block3(h, sn_update=sn_update) # (bs, 32, 32, DEPTH)
        h = self.self_atten(h, sn_update=sn_update)

        h = self.pad(h) # (bs, 34, 34, 256)
        h = self.c1(h, sn_update=sn_update)  # (bs, 31, 31, 512)

        h = self.bn(h)
        h = tf.nn.relu(h)

        h = self.pad(h)  # (bs, 33, 33, 512)
        h = self.c2(h, sn_update=sn_update)  # (bs, 30, 30, 1)
        return h