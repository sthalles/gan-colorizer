import tensorflow as tf
from layers.transpose_conv_sn import SNTransposeConv2D
from layers.sn_non_local_block import SNNonLocalBlock
from layers.orthogonal_regularization import conv_orthogonal_regularizer
from layers.resblock_down import OptimizedBlock, BlockDown
from layers.resblock_up import BlockUp

class UNetGenerator(tf.keras.Model):
    def __init__(self, ch, output_depth, activation=tf.keras.layers.ReLU()):
        super(UNetGenerator, self).__init__()

        initializer = tf.keras.initializers.Orthogonal()
        kernel_regularizer = conv_orthogonal_regularizer(0.0001)

        # encoder
        self.down1 = OptimizedBlock(ch) # 128
        self.down2 = BlockDown(ch, ch * 2, activation=activation, downsample=True) # 64
        self.down3 = BlockDown(ch * 2, ch * 2, activation=activation, downsample=True) # 32
        self.enc_attention = SNNonLocalBlock(ch * 2, initializer=initializer,
                                             kernel_regularizer=kernel_regularizer)
        self.down4 = BlockDown(ch * 2, ch * 4, activation=activation, downsample=True) # 16
        self.down5 = BlockDown(ch * 4, ch * 4, activation=activation, downsample=True) # 8
        self.down6 = BlockDown(ch * 4, ch * 8, activation=activation, downsample=True) # 4
        self.down7 = BlockDown(ch * 8, ch * 8, activation=activation, downsample=True) # 2
        self.down8 = BlockDown(ch * 8, ch * 16, activation=activation, downsample=True) # 1

        # decoder
        self.up1 = BlockUp(ch * 16, ch * 8, activation=activation, upsample=True) # 2
        self.up2 = BlockUp(ch * 8, ch * 8, activation=activation, upsample=True) # 4
        self.up3 = BlockUp(ch * 8, ch * 4, activation=activation, upsample=True) # 8
        self.up4 = BlockUp(ch * 4, ch * 4, activation=activation, upsample=True) # 16
        self.up5 = BlockUp(ch * 4, ch * 2, activation=activation, upsample=True) # 32
        self.dec_attention = SNNonLocalBlock(ch * 2, initializer=initializer,
                                             kernel_regularizer=kernel_regularizer)
        self.up6 = BlockUp(ch * 2, ch * 2, activation=activation, upsample=True) # 64
        self.up7 = BlockUp(ch * 2, ch, activation=activation, upsample=True) # 128

        self.concat = tf.keras.layers.Concatenate()

        self.conv = SNTransposeConv2D(output_depth, 4, strides=2,
                                      padding='SAME',
                                      kernel_initializer=initializer,
                                      kernel_regularizer=kernel_regularizer)

    def call(self, h, sn_update, **kwargs):
        # encoder forward
        down1 = self.down1(h, sn_update=sn_update, **kwargs)
        down2 = self.down2(down1, sn_update=sn_update, **kwargs)
        h = self.down3(down2, sn_update=sn_update, **kwargs)
        h = self.enc_attention(h, sn_update=sn_update)
        down4 = self.down4(h, sn_update=sn_update, **kwargs)
        h = self.down5(down4, sn_update=sn_update, **kwargs)
        down6 = self.down6(h, sn_update=sn_update, **kwargs)
        h = self.down7(down6, sn_update=sn_update, **kwargs)
        h = self.down8(h, sn_update=sn_update, **kwargs)

        # decoder forward
        h = self.up1(h, sn_update=sn_update, **kwargs)
        h = self.up2(h, sn_update=sn_update, **kwargs)

        h = self.concat([h, down6])

        h = self.up3(h, sn_update=sn_update, **kwargs)
        h = self.up4(h, sn_update=sn_update, **kwargs)
        h = self.concat([h, down4])

        h = self.up5(h, sn_update=sn_update, **kwargs)
        h = self.dec_attention(h, sn_update=sn_update)

        h = self.up6(h, sn_update=sn_update, **kwargs)
        h = self.concat([h, down2])

        h = self.up7(h, sn_update=sn_update, **kwargs)
        h = self.concat([h, down1])

        return tf.nn.tanh(self.conv(h, sn_update=sn_update))
