import numpy as np
import TensorFrost as tf

tf.initialize(tf.cpu)

def invert_triangular_tensorfrost(matrix, lower=True):
    n, n = matrix.shape
    inverted = tf.zeros([n, n])

    if not lower: #transpose the matrix to make it lower triangular
        matrix = matrix.T

    with tf.loop(n) as i:
        inverted[i, i] = 1.0 / matrix[i, i]
        p, k = tf.indices([i, i])
        t, = tf.indices([i])
        inverted[i, t] = -tf.sum(matrix[i, p] * inverted[p, k], axis=0) / matrix[i, i]

    if not lower: #transpose the matrix back
        inverted = inverted.T

    return inverted

def InvertUpperTriangularMatrix():
    A = tf.input([-1, -1], tf.float32)

    inverted = invert_triangular_tensorfrost(A, lower=False)
    return inverted

invert_upper_triangular = tf.compile(InvertUpperTriangularMatrix)