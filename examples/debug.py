import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.initialize(tf.opengl)

def transpose(A):
    N, M = A.shape
    i, j = tf.indices([M, N])
    return A[j, i] * 1.0

def matmul():
    A = tf.input([-1, -1], tf.float32)
    N, M = A.shape
    B = tf.input([M, -1], tf.float32)
    K = B.shape[1]
        
    Bt = transpose(B)
        
    #C = tf.zeros([N, K])
    #i, j, k = tf.indices([N, K, M])
    #tf.scatterAdd(C[i, j], A[i, k] * Bt[j, k])
        
    C = tf.buffer([N, K])
        
    i, j = C.indices
        
    s = tf.zeros([N, K], tf.float32)
    def loop_body(k):
        s.set(s + A[i, k] * Bt[j, k])
         
    tf.loop(loop_body, 0, M, 1)
        
    C[i, j] = s
        
    return [C]

mmul = tf.compile(matmul)

Anp = np.random.rand(64, 32).astype(np.float32)
Bnp = np.random.rand(32, 48).astype(np.float32)

A = tf.tensor(Anp)
B = tf.tensor(Bnp)
C, = mmul(A, B)

Cnp = C.numpy

#compare to numpy
Cnp2 = Anp @ Bnp

print(Cnp)
print(Cnp2)

Cerror = np.linalg.norm(Cnp - Cnp2) / np.linalg.norm(Cnp2)
print("Error:", Cerror)

tf.initialize(tf.opengl)

def transpose(A):
    N, M = A.shape
    i, j = tf.indices([M, N])
    return A[j, i] * 1.0

def matmul():
    A = tf.input([-1, -1], tf.float32)
    N, M = A.shape
    B = tf.input([M, -1], tf.float32)
    K = B.shape[1]
        
    Bt = transpose(B)
        
    #C = tf.zeros([N, K])
    #i, j, k = tf.indices([N, K, M])
    #tf.scatterAdd(C[i, j], A[i, k] * Bt[j, k])
        
    C = tf.buffer([N, K])
        
    i, j = C.indices
        
    s = tf.zeros([N, K], tf.float32)
    def loop_body(k):
        s.set(s + A[i, k] * Bt[j, k])
         
    tf.loop(loop_body, 0, M, 1)
        
    C[i, j] = s
        
    return [C]

mmul = tf.compile(matmul)

Anp = np.random.rand(64, 32).astype(np.float32)
Bnp = np.random.rand(32, 48).astype(np.float32)

A = tf.tensor(Anp)
B = tf.tensor(Bnp)
C, = mmul(A, B)

Cnp = C.numpy

#compare to numpy
Cnp2 = Anp @ Bnp

print(Cnp)
print(Cnp2)

Cerror = np.linalg.norm(Cnp - Cnp2) / np.linalg.norm(Cnp2)
print("Error:", Cerror)