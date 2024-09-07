import numpy as np
import TensorFrost as tf

tf.initialize(tf.opengl)

def Test():
    A = tf.input([-1], tf.float32)
    i, = A.indices

    for j in range(16):
        A = A + tf.sin(A[i - 1] + A[i + 1])

    return A

test = tf.compile(Test)

A = np.array([1, 2, 3, 4, 6], dtype=np.float32)

restf = test(A)

