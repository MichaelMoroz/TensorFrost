import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.cpu)

def BadMatrixMultiplication():
    A = tf.input([-1, -1], tf.float32)
    N, M = A.shape
    B = tf.input([M, -1], tf.float32) #M must match
    K = B.shape[1]

    C = tf.zeros([N, K])
    i, j, k = tf.indices([N, K, M])
    tf.scatterAdd(C[i, j], A[i, k] * B[k, j])

    return C

matmul = tf.compile(BadMatrixMultiplication)

Anp = np.random.rand(100, 100).astype(np.float32)
Bnp = np.random.rand(100, 100).astype(np.float32)

Atf = tf.tensor(Anp)
Btf = tf.tensor(Bnp)

Ctf = matmul(Atf, Btf)

Cnp = np.matmul(Anp, Bnp)

print(np.allclose(Ctf.numpy, Cnp))


