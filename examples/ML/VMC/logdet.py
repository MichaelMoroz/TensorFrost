import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)

def qr_decomposition_tensorfrost(A):
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
    return Q, R

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

def invert_matrix(matrix):
    Q, R = qr_decomposition_tensorfrost(matrix)
    R_inv = invert_triangular_tensorfrost(R, lower=False)
    return R_inv @ Q.T

def logdet(matrix):
    Q, R = qr_decomposition_tensorfrost(matrix)
    i = tf.index(0, [matrix.shape[0]])
    Rdiag = tf.log(tf.abs(R[i, i]))
    return tf.sum(Rdiag)

def logdet_op(inputs, tensor, axes):
    return [logdet(inputs[0])]

def logdet_op_vjp(inputs, gradient, tensor):
    return [invert_matrix(inputs[0]) * gradient]

tf.register_custom_operation("logdet", ["f_f"], logdet_op, logdet_op_vjp)

def ProgramTest():
    A = tf.input([-1, -1], tf.float32)
    B = tf.custom("logdet", [A], [])
    dB_dA = tf.grad(B, A)
    return B, dB_dA

test = tf.compile(ProgramTest)