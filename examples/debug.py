import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt

def SomeFunction():
    u = tf.input([16, 16])
    v = tf.input([16, 16])

    i = u.index(0)
    j = u.index(1)

    laplacian = u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - u[i, j] * 4.0

    dt = 0.1
    v_new = v + dt*laplacian
    u_new = u + dt*v_new

    return [u_new, v_new]

def SomeFunction2():
    A = tf.input([-1, -1])
    B = tf.input([-1, -1])

    N = A.shape[0]
    M = A.shape[1]
    K = B.shape[1]

    i = tf.index(0, [N, K, M])
    j = tf.index(1, [N, K, M])
    k = tf.index(2, [N, K, M])

    C = tf.zeros([N, K])
    tf.scatterAdd(C[i, j], A[i, k] * B[k, j])

    return [C]

def SomeFunction3():
    A = tf.input([-1, -1])
    B = tf.input([-1, -1])
    C = A + B
    return [C]


# Create a program that initializes the wave simulation
SomeFunctionProgram = tf.Program(SomeFunction2)

SomeFunctionProgram(list())

SomeFunctionProgram.ListGraphOperations()
