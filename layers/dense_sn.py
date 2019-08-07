import tensorflow as tf
from layers.ops import power_iteration


class SNDense(tf.keras.Model):
    def __init__(self, units, max_iters=1, kernel_initializer='normal', use_bias=True, kernel_regularizer=None):
        super(SNDense, self).__init__()
        self.max_iters = max_iters
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.u = None

    def build(self, input_shape):
        # kernel shape [kernel_size, kernel_size, in_channels, out_channels]
        self.hidden_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                              initializer=self.kernel_initializer,
                                              trainable=True, dtype=tf.float32,
                                              regularizer=self.kernel_regularizer,
                                              name='hidden_weights')
        if self.use_bias:
            self.bias = self.add_weight(name='bias', shape=[self.units],
                                        initializer='zeros', trainable=True)

        self.u = self.add_weight(
            'sn_estimate',
            shape=[1, self.units],
            initializer='normal',
            dtype=self.hidden_weights.dtype,
            trainable=False)
        assert self.u.shape == [1, self.units], "Wrong shape for SNDense layer"

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
        hidden_layer = self.compute_spectral_normal(self.hidden_weights, sn_update=sn_update)
        hidden = tf.matmul(x, hidden_layer)
        if self.use_bias:
            hidden = tf.nn.bias_add(hidden, self.bias)
        return hidden
