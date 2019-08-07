import tensorflow as tf
from layers.ops import power_iteration


class SNTransposeConv2D(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, strides=1, padding='SAME', max_iters=1,
                 kernel_initializer='normal', use_bias=True, kernel_regularizer=None):
        super(SNTransposeConv2D, self).__init__()
        self.max_iters = max_iters
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.filters = filters
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.kernel_regularizer = kernel_regularizer
        self.u = None

    def build(self, input_shape):
        # kernel shape [kernel_size, kernel_size, output_channels, in_channels]
        self.kernel = self.add_weight(shape=(self.kernel_size, self.kernel_size, self.filters, input_shape[-1]),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True, dtype=tf.float32,
                                      name='kernel')

        self.u = self.add_weight(
            'sn_estimate',
            shape=[1, input_shape[-1]],
            initializer='normal',
            dtype=self.kernel.dtype,
            trainable=False)
        assert self.u.shape == [1, input_shape[-1]], "Wrong shape for SNDense layer"

        if self.use_bias:
            self.bias = self.add_weight(name='biases', shape=[self.filters],
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

    def call(self, x, sn_update, output_shape=None):
        assert sn_update is not None, "sn_update parameter not provided."
        if output_shape is None:
            input_shape = x.shape
            output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, self.filters]

        w_bar = self.compute_spectral_normal(self.kernel, sn_update)
        deconv = tf.nn.conv2d_transpose(x, w_bar, output_shape=output_shape, strides=[1, self.strides, self.strides, 1],
                                        padding=self.padding)

        if self.use_bias:
            deconv = tf.nn.bias_add(deconv, self.bias)

        deconv.shape.assert_is_compatible_with(output_shape)
        return deconv
