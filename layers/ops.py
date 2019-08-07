import tensorflow as tf


def _l2normalizer(v, eps=1e-12):
    """l2 normize the input vector."""
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def power_iteration(W, u, rounds=1):
    '''
    Accroding the paper, we only need to do power iteration one time.
    '''
    u_ = u

    for i in range(rounds):
        v_ = _l2normalizer(tf.matmul(u_, W, transpose_b=True))
        u_ = _l2normalizer(tf.matmul(v_, W))

    W_sn = tf.squeeze(tf.matmul(tf.matmul(v_, W), u_, transpose_b=True))
    return W_sn, u_, v_
