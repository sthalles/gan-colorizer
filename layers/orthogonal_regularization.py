import tensorflow as tf

def conv_orthogonal_regularizer(scale):
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w):
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        _, _, _, c = w.get_shape().as_list()

        w = tf.reshape(w, [-1, c])

        # Identity matrix
        I = tf.eye(c)

        # Modified Orthogonal Regularizer proposed in the BigGan paper
        # remove the diagonal values of the identity
        reg = tf.matmul(w, w, transpose_a=True) * (1 - I)

        # calc the loss. Obs tf.nn.l2_loss computes half of the l2 loss
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg


def dense_orthogonal_regularizer(scale):
    """ Defining the Orthogonal regularizer and return the function at last to be used in Fully Connected Layer """

    def ortho_reg_fully(w):
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        _, c = w.get_shape().as_list()

        # Identity matrix
        I = tf.eye(c)

        # Modified Orthogonal Regularizer proposed in the BigGan paper
        reg = tf.matmul(a=w, b=w, transpose_a=True) * (1 - I)

        # calc the loss. Obs tf.nn.l2_loss computes half of the l2 loss
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg_fully