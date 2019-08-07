import tensorflow as tf
from layers.ops import power_iteration


class SNConv2D(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, strides=1, padding='VALID', max_iters=1,
                 kernel_initializer='normal', use_bias=True, kernel_regularizer=None):
        super(SNConv2D, self).__init__()
        self.max_iters = max_iters
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.u = None

    def build(self, input_shape):
        # kernel shape [kernel_size, kernel_size, in_channels, out_channels]
        self.kernel = self.add_weight(shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.filters),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      name='kernel')
        self.u = self.add_weight(
            'sn_estimate',
            shape=[1, self.filters],
            initializer='normal',
            dtype=self.kernel.dtype,
            trainable=False)

        assert(self.u.shape == (1,self.filters))

        if self.use_bias:
            self.biases = self.add_weight(name='biases', shape=[self.filters],
                                          initializer='zeros', trainable=True)

    def compute_spectral_normal(self, weights, sn_update):
        # Spectrally Normalized Weight
        W_shape = weights.shape.as_list()

        # Flatten the Tensor
        W_mat = tf.reshape(weights, [-1, W_shape[-1]])  # [-1, output_channel]
        W_sn, u, v = power_iteration(W_mat, self.u, rounds=self.max_iters)

        if sn_update == True:
            # Update estimated 1st singular vector
            self.u.assign(u)

        W_mat = W_mat / W_sn
        w_bar = tf.reshape(W_mat, W_shape)

        return w_bar

    def call(self, x, sn_update):
        assert sn_update is not None, "sn_update parameter not provided."
        w_bar = self.compute_spectral_normal(self.kernel, sn_update)
        conv = tf.nn.conv2d(x, w_bar, strides=[1, self.strides, self.strides, 1], padding=self.padding)
        if self.use_bias:
            conv = tf.nn.bias_add(conv, self.biases)
        return conv
