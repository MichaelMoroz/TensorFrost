# %%
import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)
def autodiff_test():
    A = tf.input([-1])
    B = tf.input(A.shape)
    C = 2.0 * B / (A ** 2.0) + tf.sin(A)
    G = tf.grad(C, A)
    return [G]

nbody = tf.compile(autodiff_test)