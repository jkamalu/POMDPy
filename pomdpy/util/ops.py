import tensorflow as tf
import numpy as np


def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_size],
                            initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn is not None:
            return activation_fn(out), w, b
        else:
            return out, w, b


def select_action(belief, vector_set):
    """
    Compute optimal action given a belief distribution
    :param belief: dim(belief) == dim(AlphaVector)
    :param vector_set
    :return: optimal action, V(b)
    """
    assert not len(vector_set) == 0

    max_v = tf.constant([-np.inf])
    best_action = tf.constant([-1])
    for av in vector_set:
        v = tf.reduce_sum(tf.mul(av.v, belief))
        best_action = tf.cond(tf.greater(v, max_v)[0], lambda: tf.constant([av.action]),
                              lambda: best_action, name='optimal_action')
        max_v = tf.maximum(v, max_v, name='V_b')

    return best_action, max_v

