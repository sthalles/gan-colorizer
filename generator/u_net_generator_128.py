import tensorflow as tf
from layers.transpose_conv_sn import SNTransposeConv2D
from layers.sn_non_local_block import SNNonLocalBlock
from layers.orthogonal_regularization import conv_orthogonal_regularizer, dense_orthogonal_regularizer
from generator.gen_blocks import UpSample, DownSample


class UNetGenerator(tf.keras.Model):
    def __init__(self, ch, output_depth, activation=tf.keras.layers.ReLU()):
        super(UNetGenerator, self).__init__()

        initializer = tf.keras.initializers.Orthogonal()
        kernel_regularizer = conv_orthogonal_regularizer(0.0001)

        # encoder
        self.down1 = DownSample(ch * 2, 3, apply_batchnorm=False, activation=activation, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)
        self.down2 = DownSample(ch * 4, 3, activation=activation, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)
        self.enc_attention = SNNonLocalBlock(ch * 4, initializer=initializer,
                                             kernel_regularizer=kernel_regularizer)
        self.down3 = DownSample(ch * 4, 3, activation=activation, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)
        self.down4 = DownSample(ch * 8, 3, activation=activation, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)
        self.down5 = DownSample(ch * 8, 3, activation=activation, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)
        self.down6 = DownSample(ch * 16, 3, activation=activation, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)
        self.down7 = DownSample(ch * 16, 3, activation=activation, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)

        # decoder
        self.up1 = UpSample(ch * 16, 3, apply_dropout=True, activation=activation, initializer=initializer,
                            kernel_regularizer=kernel_regularizer)
        self.up2 = UpSample(ch * 8, 3, apply_dropout=True, activation=activation, initializer=initializer,
                            kernel_regularizer=kernel_regularizer)
        self.up3 = UpSample(ch * 8, 3, apply_dropout=True, activation=activation, initializer=initializer,
                            kernel_regularizer=kernel_regularizer)
        self.up4 = UpSample(ch * 4, 3, activation=activation, initializer=initializer,
                            kernel_regularizer=kernel_regularizer)
        self.up5 = UpSample(ch * 4, 3, activation=activation, initializer=initializer,
                            kernel_regularizer=kernel_regularizer)
        self.dec_attention = SNNonLocalBlock(ch * 8, initializer=initializer,
                                             kernel_regularizer=kernel_regularizer)
        self.up6 = UpSample(ch * 2, 3, activation=activation, initializer=initializer,
                            kernel_regularizer=kernel_regularizer)

        self.concat = tf.keras.layers.Concatenate()

        self.conv = SNTransposeConv2D(output_depth, 4, strides=2,
                                      padding='SAME',
                                      kernel_initializer=initializer,
                                      kernel_regularizer=kernel_regularizer)

    def call(self, x, sn_update, **kwargs):
        # encoder forward
        down1 = self.down1(x, sn_update=sn_update, **kwargs)
        down2 = self.down2(down1, sn_update=sn_update, **kwargs)
        enc_attention = self.enc_attention(down2, sn_update=sn_update)
        down3 = self.down3(enc_attention, sn_update=sn_update, **kwargs)
        down4 = self.down4(down3, sn_update=sn_update, **kwargs)
        down5 = self.down5(down4, sn_update=sn_update, **kwargs)
        down6 = self.down6(down5, sn_update=sn_update, **kwargs)
        down7 = self.down7(down6, sn_update=sn_update, **kwargs)

        # decoder forward
        up1 = self.up1(down7, sn_update=sn_update, **kwargs)
        up1 = self.concat([up1, down6])

        up2 = self.up2(up1, sn_update=sn_update, **kwargs)
        up2 = self.concat([up2, down5])

        up3 = self.up3(up2, sn_update=sn_update, **kwargs)
        up3 = self.concat([up3, down4])

        up4 = self.up4(up3, sn_update=sn_update, **kwargs)
        up4 = self.concat([up4, down3])

        up5 = self.up5(up4, sn_update=sn_update, **kwargs)
        up5 = self.concat([up5, down2])

        dec_attention = self.dec_attention(up5, sn_update=sn_update)

        up6 = self.up6(dec_attention, sn_update=sn_update, **kwargs)
        up6 = self.concat([up6, down1])

        return tf.nn.tanh(self.conv(up6, sn_update=sn_update))
