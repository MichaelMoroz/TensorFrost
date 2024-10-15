import numpy as np
import TensorFrost as tf

tf.initialize(tf.opengl)

def Test():
    A = tf.input([-1, 3], tf.float32)

    with tf.kernel([A.shape[0], 1]) as (i, _):
        i, j = tf.indices([A.shape[0], 3])
        A[i, j] = A[i, j] + 1.0

        # v = tf.local_buffer(3, tf.float32)
        #
        # with tf.loop(3) as j:
        #     tf.local_store(v, j, A[i, j])
        #
        # with tf.loop(3) as j:
        #     A[i, j] = tf.local_load(v, j) + 1.0

    return A

test = tf.compile(Test)

A = np.random.rand(10, 3).astype(np.float32)

restf = test(A)