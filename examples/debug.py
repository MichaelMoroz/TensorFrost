# %%
import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)

dt = 0.0001

def Force(dx, dv, rho):
    i, j, k = dx.indices
    dist = tf.unsqueeze(tf.norm(dx))
    dvdotdx = tf.unsqueeze(tf.dot(dv, dx))
    weight = tf.exp(-dist / 0.015)
    pressure = 0.2 * (rho[i] + rho[j])**2.0 * weight
    Fij = 0.2 * dx / (dist ** 3.0 + 5e-3) + 100.0 * dvdotdx * dx * weight / (dist*dist + 1e-5)  - 50.0 * pressure * dx / (dist*dist + 1e-5)
    Fij = tf.select(i == j, 0.0, Fij)
    return Fij

def Density(dx):
    dist = tf.norm(dx)
    weight = tf.exp(-dist / 0.015)
    rho = 1.0 * weight
    return rho

def n_body():
    X = tf.input([-1, 3], tf.float32)
    N = X.shape[0]
    V = tf.input([N, 3], tf.float32)

    i, j, k = tf.indices([N, N, 3])
    dx = X[i,k] - X[j,k]
    dv = V[i,k] - V[j,k]

    rho = tf.sum(Density(dx), axis=1)
    Fi = tf.sum(Force(dx, dv, rho), axis=1)

    Vnew = V + Fi * dt
    Xnew = X + Vnew * dt

    return [Xnew, Vnew]

nbody = tf.compile(n_body)