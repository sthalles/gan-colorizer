import tensorflow as tf
from tensorflow.python.keras.layers import MaxPool2D
from layers.conv_sn import SNConv2D


class SNNonLocalBlock(tf.keras.layers.Layer):
    def __init__(self, n_channels, initializer=None, kernel_regularizer=None):
        super(SNNonLocalBlock, self).__init__()

        self.theta = SNConv2D(n_channels // 8, 1,
                              strides=1, padding='SAME',
                              kernel_initializer=initializer,
                              use_bias=False, kernel_regularizer=kernel_regularizer)

        self.phi = SNConv2D(n_channels // 8, 1,
                            strides=1, padding='SAME',
                            kernel_initializer=initializer,
                            use_bias=False, kernel_regularizer=kernel_regularizer)

        self.max_pool = MaxPool2D(pool_size=2, strides=2)

        self.g = SNConv2D(n_channels // 2, 1,
                          strides=1, padding='SAME',
                          kernel_initializer=initializer,
                          use_bias=False, kernel_regularizer=kernel_regularizer)

        self.sigma = self.add_weight(shape=(),
                                     name="sigma",
                                     initializer='zeros',
                                     trainable=True)

        self.conv = SNConv2D(filters=n_channels, kernel_size=1, padding='VALID',
                             kernel_initializer=initializer, strides=1, kernel_regularizer=kernel_regularizer)

    def call(self, x, sn_update):
        # get the input shape
        batch_size, h, w, num_channels = x.shape

        location_num = h * w
        downsampled_num = (h // 2) * (w // 2)

        # theta path
        theta = self.theta(x, sn_update=sn_update)
        theta = tf.reshape(theta, shape=[batch_size, location_num, num_channels // 8])

        # phi path
        phi = self.phi(x, sn_update=sn_update)
        phi = self.max_pool(phi)
        phi = tf.reshape(phi, shape=[batch_size, downsampled_num, num_channels // 8])

        attn_map = tf.matmul(theta, phi, transpose_b=True)
        # The softmax operation is performed on each row
        attn_map = tf.nn.softmax(attn_map, axis=-1)

        # g path
        g = self.g(x, sn_update=sn_update)
        g = self.max_pool(g)

        g = tf.reshape(g, shape=[batch_size, downsampled_num, num_channels // 2])
        attn_g = tf.matmul(attn_map, g)
        attn_g = tf.reshape(attn_g, shape=[batch_size, h, w, num_channels // 2])

        attn_g = self.conv(attn_g, sn_update=sn_update)
        return x + self.sigma * attn_g
