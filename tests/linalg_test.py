import numpy as np
import TensorFrost as tf
import unittest

tf.initialize(tf.cpu)

def modified_gram_schmidt(A):
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for i in range(n-1):
        R[i, i] = np.linalg.norm(A[:, i])
        Q[:, i] = A[:, i] / R[i, i]
        R[i, i+1:n] = np.dot(Q[:, i].T, A[:, i+1:n])
        A[:, i+1:n] -= np.outer(Q[:, i], R[i, i+1:n])
    R[n-1, n-1] = np.linalg.norm(A[:, n-1])
    Q[:, n-1] = A[:, n-1] / R[n-1, n-1]
    return Q, R

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

    inverted[0, 0] = 1.0 / matrix[0, 0]
    
    with tf.loop(1, n) as i:
        inverted[i, i] = 1.0 / matrix[i, i]
        p, k = tf.indices([i, i])
        t, = tf.indices([i])
        inverted[i, t] = -tf.sum(matrix[i, p] * inverted[p, k], axis=0) / matrix[i, i]

    if not lower: #transpose the matrix back
        inverted = inverted.T

    return inverted

def InvertMatrix():
    A = tf.input([-1, -1], tf.float32)

    Q, R = qr_decomposition_tensorfrost(A)
    R_inv = invert_triangular_tensorfrost(R, lower=False)
    A_inv = R_inv @ Q.T

    return Q, R, R_inv, A_inv

class TestQRInversion(unittest.TestCase):
    def test_qr_inversion(self):
        #compile the program
        invert_matrix = tf.compile(InvertMatrix)

        A = np.random.rand(5, 5)
        Atf = tf.tensor(A)
        Qtf, Rtf, Rinvtf, Ainvtf = invert_matrix(Atf)
        Qnp = Qtf.numpy
        Rnp = Rtf.numpy
        Rinvtf = Rinvtf.numpy
        Ainvtf = Ainvtf.numpy

        Q, R = modified_gram_schmidt(A)
        Rinv = np.linalg.inv(R)
        Ainv = np.linalg.inv(A)

        self.assertTrue(np.allclose(A, np.dot(Q, R)))
        self.assertTrue(np.allclose(A, np.dot(Qnp, Rnp)))
        #inverse will be less accurate due to the nature of the algorithm
        self.assertTrue(np.allclose(Ainv, Ainvtf, atol=1e-3))
        self.assertTrue(np.allclose(Rinv, Rinvtf, atol=1e-3))