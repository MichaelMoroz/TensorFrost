# %%
import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)
def autodiff_test():
    A = tf.input([-1, -1])
    N, M = A.shape
    B = tf.input([M, -1])

    C = tf.matmul(A, B)

    G = tf.grad(C, A)

    return [G]

nbody = tf.compile(autodiff_test)
