import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.initialize(tf.opengl)

#dynamic size QR decomposition
#def QRDecomposition():
#    A = tf.input([-1, -1], tf.float32)
#
#    m, n = A.shape
#    Q = tf.zeros([m, n])
#    R = tf.zeros([n, n])
#    j = tf.index(0, [m])
#
#    def loop_body(i):
#        R[i, i] = tf.norm(A[j, i])
#        Q[j, i] = A[j, i] / R[i, i]
#
#        p, k = tf.index_grid([0, i + 1], [m, n])
#        t, = tf.index_grid([i+1], [n])
#        R[i, t] = tf.sum(Q[p, i] * A[p, k], axis=0)
#        A[p, k] -= Q[p, i] * R[i, k]
#
#    tf.loop(loop_body, 0, n-1, 1)
#
#    R[n-1, n-1] = tf.norm(A[j, n-1])
#    Q[j, n-1] = A[j, n-1] / R[n-1, n-1]
#
#    return [Q, R]
#
#qr = tf.compile(QRDecomposition)
#
##generate random matrix
#A = np.random.rand(5, 5)
#
##compute QR decomposition using TensorFrost
#Atf = tf.tensor(A)
#Qtf, Rtf = qr(Atf)
#Qnp = Qtf.numpy
#Rnp = Rtf.numpy
#
##check if QR decomposition is correct
#print("QR decomposition using TensorFrost is correct:", np.allclose(A, np.dot(Qnp, Rnp)))
#
##check error
#print("Error using TensorFrost:", np.linalg.norm(A - np.dot(Qnp, Rnp)))
#
##print Q and R
#print("Q:\n", Qnp)
#print("R:\n", Rnp)

def MaxBlock():
	A = tf.input([-1, -1, -1, -1], tf.float32)
	N, Bx, By, Bz = A.shape
	Ar = tf.reshape(A, [N, Bx*By*Bz])
	#only reduces one dimension, by default it is the last dimension
	max_val = tf.max(Ar)
	min_val = tf.min(Ar)
	sum = tf.sum(Ar)
	mean = tf.mean(Ar)
	norm = tf.norm(Ar)
	total_max = tf.max(max_val)
	total_min = tf.min(min_val)
	return [max_val, min_val, sum, mean, norm, total_max, total_min]

max_block = tf.compile(MaxBlock)
