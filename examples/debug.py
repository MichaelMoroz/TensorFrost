import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.initialize(tf.opengl)

#dynamic size QR decomposition
def QRDecomposition():
    A = tf.input([-1, -1], tf.float32)

    m, n = A.shape
    Q = tf.zeros([m, n])
    R = tf.zeros([n, n])
    j = tf.index(0, [m])

    def loop_body(i):
        R[i, i] = tf.norm(A[j, i])
        Q[j, i] = A[j, i] / R[i, i]

        p, k = tf.index_grid([0, i + 1], [m, n])
        t, = tf.index_grid([i+1], [n])
        R[i, t] = tf.sum(Q[p, i] * A[p, k], axis=0)
        A[p, k] -= Q[p, i] * R[i, k]

    tf.loop(loop_body, 0, n-1, 1)

    R[n-1, n-1] = tf.norm(A[j, n-1])
    Q[j, n-1] = A[j, n-1] / R[n-1, n-1]

    return [Q, R]


#def QRDecomposition():
#    A = tf.input([-1, -1], tf.float32)
#    m, n = A.shape
#
#    Q = tf.zeros([m, n])
#    R = tf.zeros([n, n])
#    def loop_body(i):
#        R[i, i] = tf.norm(A[:, i])
#        Q[:, i] = A[:, i] / R[i, i]
#        R[i, i+1:n] = tf.sum(Q[:, i] * A[:, i+1:n], axis=0)
#        A[:, i+1:n] -= Q[:, i] * R[i, i+1:n]
#
#    tf.loop(loop_body, 0, n-1, 1)
#    R[n-1, n-1] = tf.norm(A[:, n-1])
#    Q[:, n-1] = A[:, n-1] / R[n-1, n-1]
#
#    return [Q, R]

qr = tf.compile(QRDecomposition)

#generate random matrix
A = np.random.rand(5, 5)

#compute QR decomposition using TensorFrost
Atf = tf.tensor(A)
Qtf, Rtf = qr(Atf)
Qnp = Qtf.numpy
Rnp = Rtf.numpy

#check if QR decomposition is correct
print("QR decomposition using TensorFrost is correct:", np.allclose(A, np.dot(Qnp, Rnp)))

#check error
print("Error using TensorFrost:", np.linalg.norm(A - np.dot(Qnp, Rnp)))

#print Q and R
print("Q:\n", Qnp)
print("R:\n", Rnp)

#def BlockMax():
#	blocks = tf.input([-1, -1, -1, -1, -1], tf.float32)
#	N, Bx, By, Bz, CH = blocks.shape
#
#	block_max = tf.max(tf.abs(tf.reshape(blocks, [N, Bx*By*Bz, CH])), axis=1)
#
#	return [block_max]
#
#bmax = tf.compile(BlockMax)
#
##generate random blocks
#blocks = np.random.rand(32, 8, 8, 8, 3)
#
##compute block max using TensorFrost
#blockstf = tf.tensor(blocks)
#block_max_tf, = bmax(blockstf)
#block_max_np = block_max_tf.numpy
#
##check if block max is correct
#print("Block max using TensorFrost is correct:", np.allclose(np.max(np.abs(blocks), axis=(1, 2, 3)), block_max_np))

