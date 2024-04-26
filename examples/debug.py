import TensorFrost as tf
import numpy as np
import time

tf.initialize(tf.opengl)

#dynamic size QR decomposition
def QRDecomposition():
    A = tf.input([-1, -1], tf.float32)

    m, n = A.shape
    Q = tf.zeros([m, n])
    R = tf.zeros([n, n])
    j = tf.index(0, [m])

    with tf.loop(n-1) as i:
        R[i, i] = tf.norm(A[j, i])
        Q[j, i] = A[j, i] / R[i, i]

        p, k = tf.index_grid([0, i + 1], [m, n])
        t, = tf.index_grid([i+1], [n])
        R[i, t] = tf.sum(Q[p, i] * A[p, k], axis=0)
        A[p, k] -= Q[p, i] * R[i, k]

    R[n-1, n-1] = tf.norm(A[j, n-1])
    Q[j, n-1] = A[j, n-1] / R[n-1, n-1]

    return [Q, R]

qr = tf.compile(QRDecomposition)