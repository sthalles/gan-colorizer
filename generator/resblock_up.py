import tensorflow as tf
from layers.conv_sn import SNConv2D
from layers.orthogonal_regularization import conv_orthogonal_regularizer

class BlockUp(tf.keras.Model):
    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3, padding='SAME',
                 activation=tf.keras.layers.ReLU(), upsample=False):
        super(BlockUp, self).__init__()
        initializer = tf.keras.initializers.Orthogonal()
        kernel_regularizer = conv_orthogonal_regularizer(0.0001)

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        self.unpooling_2d = tf.keras.layers.UpSampling2D()

        self.c1 = SNConv2D(hidden_channels, kernel_size=kernel_size, padding=padding,
                           kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)

        self.c2 = SNConv2D(out_channels, kernel_size=kernel_size, padding=padding,
                           kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)

        self.b1 = tf.keras.layers.BatchNormalization()
        self.b2 = tf.keras.layers.BatchNormalization()

        if self.learnable_sc:
            self.c_sc = SNConv2D(out_channels, kernel_size=1, padding="VALID",
                                 kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)

    def residual(self, x, sn_update, **kwargs):
        assert sn_update is not None, "Specify the 'sn_update' parameter"
        h = x
        h = self.b1(h, **kwargs)
        h = self.activation(h)

        if self.upsample:
            h = self.unpooling_2d(h)

        h = self.c1(h, sn_update=sn_update)

        h = self.b2(h, **kwargs)
        h = self.activation(h)
        h = self.c2(h, sn_update=sn_update)
        return h

    def shortcut(self, x, sn_update):
        if self.learnable_sc:

            if self.upsample:
                x = self.unpooling_2d(x)

            x = self.c_sc(x, sn_update=sn_update)
            return x
        else:
            return x

    def __call__(self, x, sn_update=None, **kwargs):
        return self.residual(x, sn_update=sn_update, **kwargs) + self.shortcut(x, sn_update=sn_update)
