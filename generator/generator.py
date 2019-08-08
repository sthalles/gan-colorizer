import tensorflow as tf
from layers.blocks import DownSample, UpSample
from layers.orthogonal_regularization import dense_orthogonal_regularizer, conv_orthogonal_regularizer


class UNetGenerator(tf.keras.Model):
    def __init__(self, ch, output_depth):
        super(UNetGenerator, self).__init__()

        initializer = tf.keras.initializers.Orthogonal()
        kernel_regularizer = conv_orthogonal_regularizer(0.0001)

        self.down1 = DownSample(ch, 4, initializer=initializer, kernel_regularizer=kernel_regularizer,
                                apply_batchnorm=False)  # (bs, 128, 128, 64)
        self.down2 = DownSample(ch * 2, 4, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)  # (bs, 64, 64, 128)
        self.down3 = DownSample(ch * 4, 4, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)  # (bs, 32, 32, 256)
        self.down4 = DownSample(ch * 8, 4, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)  # (bs, 16, 16, 512)
        self.down5 = DownSample(ch * 8, 4, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)  # (bs, 8, 8, 512)
        self.down6 = DownSample(ch * 8, 4, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)  # (bs, 8, 8, 512)
        self.down7 = DownSample(ch * 8, 4, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)  # (bs, 8, 8, 512)
        self.down8 = DownSample(ch * 8, 4, initializer=initializer,
                                kernel_regularizer=kernel_regularizer)  # (bs, 8, 8, 512)

        self.up1 = UpSample(ch * 8, 4, initializer=initializer, kernel_regularizer=kernel_regularizer,
                            apply_dropout=True)  # (bs, 2, 2, 1024)
        self.up2 = UpSample(ch * 8, 4, initializer=initializer, kernel_regularizer=kernel_regularizer,
                            apply_dropout=True)  # (bs, 4, 4, 1024)
        self.up3 = UpSample(ch * 8, 4, initializer=initializer, kernel_regularizer=kernel_regularizer,
                            apply_dropout=True)  # (bs, 8, 8, 1024)
        self.up4 = UpSample(ch * 8, 4, initializer=initializer,
                            kernel_regularizer=kernel_regularizer)  # (bs, 16, 16, 1024)
        self.up5 = UpSample(ch * 4, 4, initializer=initializer, kernel_regularizer=kernel_regularizer)  # (bs, 32, 32, 512)
        self.up6 = UpSample(ch * 2, 4, initializer=initializer, kernel_regularizer=kernel_regularizer)  # (bs, 64, 64, 256)
        self.up7 = UpSample(ch, 4, initializer=initializer,
                            kernel_regularizer=kernel_regularizer)  # (bs, 128, 128, 128)

        self.last = tf.keras.layers.Conv2DTranspose(output_depth, 4,
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer,
                                                    kernel_regularizer=kernel_regularizer,
                                                    activation='tanh')  # (bs, 256, 256, 3)

        self.concat = tf.keras.layers.Concatenate()

    def call(self, h, sn_update, **kwargs):
        down1 = self.down1(h, sn_update=sn_update, **kwargs)
        down2 = self.down2(down1, sn_update=sn_update, **kwargs)
        down3 = self.down3(down2, sn_update=sn_update, **kwargs)
        down4 = self.down4(down3, sn_update=sn_update, **kwargs)
        down5 = self.down5(down4, sn_update=sn_update, **kwargs)
        down6 = self.down6(down5, sn_update=sn_update, **kwargs)
        down7 = self.down7(down6, sn_update=sn_update, **kwargs)
        down8 = self.down8(down7, sn_update=sn_update, **kwargs)

        h = self.up1(down8, sn_update=sn_update, **kwargs)
        h = self.concat([h, down7])

        h = self.up2(h, sn_update=sn_update, **kwargs)
        h = self.concat([h, down6])

        h = self.up3(h, sn_update=sn_update, **kwargs)
        h = self.concat([h, down5])

        h = self.up4(h, sn_update=sn_update, **kwargs)
        h = self.concat([h, down4])

        h = self.up5(h, sn_update=sn_update, **kwargs)
        h = self.concat([h, down3])

        h = self.up6(h, sn_update=sn_update, **kwargs)
        h = self.concat([h, down2])

        h = self.up7(h, sn_update=sn_update, **kwargs)
        h = self.concat([h, down1])

        h = self.last(h)
        return h
