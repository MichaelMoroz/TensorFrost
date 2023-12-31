import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.initialize(tf.cpu, "/Zi")
#tf.initialize(tf.cpu, "/O2 /fp:fast /openmp:experimental /Zi")

#def test():
#    canvas = tf.buffer([32, 32, 3], tf.float32)
#
#    i,j = tf.indices([32, 32])
#    x, y = tf.float(i), tf.float(j)
#    x, y = x/32.0, y/32.0
#
#    vx = tf.sin(2.0*3.141592*x)
#    vy = tf.sin(2.0*3.141592*y)
#    mag = 0.5*tf.sqrt(vx*vx + vy*vy)
#
#    mag = tf.clamp(mag, 0.0, 1.0)
#    canvas[i, j, 0] = (0.277 + mag * (0.105 + mag * (-0.330 + mag * (-4.634 + mag * (6.228 + mag * (4.776 - 5.435 * mag))))))
#    canvas[i, j, 1] = (0.005 + mag * (1.404 + mag * (0.214 + mag * (-5.799 + mag * (14.179 + mag * (-13.745 + 4.645 * mag))))))
#    canvas[i, j, 2] = (0.334 + mag * (1.384 + mag * (0.095 + mag * (-19.332 + mag * (56.690 + mag * (-65.353 + 26.312 * mag))))))
#
#    a, = tf.indices([16])
#    canvas[a+8, 8, 0] = 1.0
#    canvas[8, a+8, 0] = 1.0
#    canvas[a+8, 24, 0] = 1.0
#    canvas[24, a+8, 0] = 1.0
#    return [canvas]
#
#
#t1 = tf.compile(test)
#
#res, = t1()
#resnp = res.numpy
#print(resnp.shape)
#print(resnp)
#
#N = 256
#M = 512
#
#def WaveEq():
#    u = tf.input([-1,-1], tf.float32)
#    v = tf.input(u.shape, tf.float32)
#
#    i,j = u.indices
#    laplacian = u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - u[i,j] * 4.0
#    force = laplacian - 0.1 * tf.sin(2.0*np.pi*u)
#    dt = 0.1
#    v_new = v + dt*force
#    u_new = u + dt*v_new
#
#    return [u_new, v_new]
#
#wave = tf.compile(WaveEq)
#
#def transpose(A):
#    N, M = A.shape
#    i, j = tf.indices([M, N])
#    return A[j, i] * 1.0
#
#def matmul():
#    A = tf.input([-1, -1], tf.float32)
#    N, M = A.shape
#    B = tf.input([M, -1], tf.float32)
#    K = B.shape[1]
#
#    Bt = transpose(B)
#
#    C = tf.zeros([N, K])
#    i, j, k = tf.indices([N, K, M])
#    tf.scatterAdd(C[i, j], A[i, k] * Bt[j, k])
#
#    return [C]
#
#mmul = tf.compile(matmul)
#print(mmul.list_operations())
#
#def test():
#    canvas = tf.zeros([8, 8], tf.float32)
#    i, j = tf.index_grid([0, 0], [8, 8], [2, 2])
#    canvas[i, j] = 1.0
#    return [canvas]
#
#t1 = tf.compile(test)
#print(t1.list_operations(compact=False))
#
#res, = t1()
#resnp = res.numpy
#print(resnp)

QRS = 64

def modified_gram_schmidt(A):
    """
    Implements the Modified Gram-Schmidt orthogonalization to get the QR decomposition of matrix A.
    A = QR
    """
    A = A.astype(float)  # Ensure A is of float type
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

def sum(A):
    n, m = A.shape
    sum_buf = tf.zeros([m], tf.float32)
    i, j = A.indices
    tf.scatterAdd(sum_buf[j], A[i, j])
    return sum_buf

def norm(A):
    A = A * 1.0
    sum_buf = tf.zeros([1], tf.float32)
    ids = tf.indices(A.shape)
    tf.scatterAdd(sum_buf[0], A[ids] ** 2)
    return tf.sqrt(sum_buf)

def QRDecomposition():
    A = tf.input([QRS, QRS], tf.float32)

    m, n = A.shape
    Q = tf.zeros([m, n])
    R = tf.zeros([n, n])

    j = tf.index(0, [m])
    for i in range(QRS-1):
        R[i, i] = norm(A[j, i])
        Q[j, i] = A[j, i] / R[i, i]

        #R[i, i+1:n] = np.dot(Q[:, i].T, A[:, i+1:n])
        #A[:, i+1:n] -= np.outer(Q[:, i], R[i, i+1:n])

        t, = tf.index_grid([i+1], [n])
        p, k = tf.index_grid([0, i+1], [m, n])
        R[i, t] = sum(Q[p, i] * A[p, k])
        A[p, k] -= Q[p, i] * R[i, k]

    R[n-1, n-1] = norm(A[j, n-1])
    Q[j, n-1] = A[j, n-1] / R[n-1, n-1]

    return [Q, R]

qr = tf.compile(QRDecomposition)
#print(qr.list_operations())

Anp = np.random.rand(QRS, QRS).astype(np.float32)
Qnp, Rnp = modified_gram_schmidt(Anp)
print(Qnp)
print(Rnp)

A = tf.tensor(Anp)
Qtf, Rtf = qr(A)
Qerror = np.linalg.norm(Qtf.numpy - Qnp) / np.linalg.norm(Qnp)
Rerror = np.linalg.norm(Rtf.numpy - Rnp) / np.linalg.norm(Rnp)
print("Q error: ", Qerror)
print("R error: ", Rerror)
if Qerror > 1e-5 or Rerror > 1e-5:
	print("QR decomposition failed")
	exit(1)