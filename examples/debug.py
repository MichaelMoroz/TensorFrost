import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.initialize(tf.cpu, "/Zi")
#tf.initialize(tf.cpu, "/O2 /fp:fast /openmp:experimental /Zi")

N = 512 
M = 512

def Jacobi(pressure, div, iterations):
    i, j = pressure.indices

    # pressure solve
    for it in range(iterations):
        pressure = (pressure[i-1, j] + pressure[i+1, j] + pressure[i, j-1] + pressure[i, j+1] - div) / 4.0

    return pressure

def Restrict(field):
    N1, M1 = field.shape
    N2, M2 = N1/2, M1/2
    i, j = tf.indices([N2, M2])
    i, j = 2*i, 2*j
    return (field[i, j] + field[i+1, j] + field[i, j+1] + field[i+1, j+1])

def Prolong(field, orig):
    i, j = orig.indices
    i, j = i/2, j/2
    return orig + field[i, j]

def Residual(pressure, div):
    i, j = pressure.indices
    return div - (pressure[i-1, j] + pressure[i+1, j] + pressure[i, j-1] + pressure[i, j+1] - 4.0*pressure)

def TEST():
    pressure = tf.input([N, M], tf.float32)
    div = tf.input([N, M], tf.float32)

    res = Residual(pressure, div)
    res = Restrict(res)
    pressure0 = Jacobi(tf.zeros(res.shape), 4.0*res, 2)

    res1 = Residual(pressure0, 4.0*res)
    res1 = Restrict(res1)
    pressure1 = Jacobi(tf.zeros(res1.shape), 16.0*res1, 1)

    return [pressure1]

test = tf.program(TEST)
test.list_operations(compact=True)

Anp = np.random.rand(N, M).astype(np.float32)
Bnp = np.random.rand(N, M).astype(np.float32)

A = tf.memory(Anp)
B = tf.memory(Bnp)

C = test(A, B)