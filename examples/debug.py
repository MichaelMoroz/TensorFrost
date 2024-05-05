# %%
import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)
def n_body():
    X = tf.input([-1, 3], tf.float32)
    N = X.shape[0]
    V = tf.input([N, 3], tf.float32)
    params = tf.input([-1], tf.float32)

    sph_rad = params[0] # 0.015
    rest_density = params[1] # 0.5
    stiffness = params[2] # 20.0
    viscosity = params[3] # 100.0
    gravity = params[4] # 1.5
    time_step = params[5] # 0.0001

    i, j, k = tf.indices([N, N, 3])
    dx = X[j,k] - X[i,k]
    dv = V[j,k] - V[i,k]

    # Compute the SPH density
    dist = tf.norm(dx)
    weight = tf.exp(-dist / sph_rad)
    rho = 1.0 * weight
    rho = tf.sum(rho, axis=1)

    # Compute the SPH forces
    dist = tf.unsqueeze(tf.norm(dx))
    dvdotdx = tf.unsqueeze(tf.dot(dv, dx))
    weight = tf.exp(-(dist / sph_rad)**2.0)
    pressure = ((rho[i] + rho[j])*0.5 - rest_density)
    Fg = gravity * dx / (dist ** 3.0 + 5e-3)
    Fvisc = viscosity * dvdotdx * dx * weight / (dist*dist + 1e-5)
    Fsph = - stiffness * pressure * dx * weight / (dist*dist + 1e-5)
    Fij = Fsph + Fvisc + Fg
    Fij = tf.select(i == j, 0.0, Fij)
    Fi = tf.sum(Fij, axis=1)

    Vnew = V + Fi * time_step
    Xnew = X + Vnew * time_step

    return [Xnew, Vnew]

nbody = tf.compile(n_body)