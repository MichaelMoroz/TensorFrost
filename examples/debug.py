import numpy as np
import TensorFrost as tf

tf.initialize(tf.opengl)

def Test():
    A = tf.input([-1, 3], tf.float32)

    with tf.kernel([A.shape[0], 3]) as (i, _):
        print(i)
        i, j = tf.indices([A.shape[0], 3])
        print(i)
        A[i, j] = A[i, j] + 1.0

    return A

test = tf.compile(Test)

A = np.random.rand(10, 3).astype(np.float32)

restf = test(A)