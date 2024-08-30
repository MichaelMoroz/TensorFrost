import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.initialize(tf.cpu)

def matmul():
    A = tf.input([-1, -1], tf.float32)
    N, M = A.shape
    B = tf.input([-1,  M], tf.float32)
    K = B.shape[1]

    C = (tf.sin(A) @ tf.cos(B.T))**2.0

    return C

matmulprog = tf.compile(matmul)