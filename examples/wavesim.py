import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt

def InitWaveSimulation(inputs):
    Xdim = 2048
    Ydim = 2048

    ids = tf.indices([Xdim, Ydim])
    i = ids[0]
    j = ids[1]

    x = tf.f32(i) / Xdim - 0.5
    y = tf.f32(j) / Ydim - 0.5

    u = tf.exp(-10000.0 * (x * x + y * y))
    v = tf.zeros([Xdim, Ydim])

    return [u, v]

def StepWaveSimulation(inputs):
    u = inputs[0]
    v = inputs[1]

    Xdim = u.shape[0]
    Ydim = u.shape[1]

    ids = tf.indices([Xdim, Ydim])
    i = ids[0]
    j = ids[1]

    u_laplacian = u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - u * 4.0

    #boundary conditions
    edge = 1.0 - tf.f32((i == 0) | (i == Xdim - 1) | (j == 0) | (j == Ydim - 1))

    #verlet integration
    dt = 0.5
    v_new = (v + u_laplacian * dt) * edge
    u_new = (u + v_new * dt) * edge

    return [u_new, v_new]

tf.backend("cpu")

# Create a program that initializes the wave simulation
InitWaveSimulationProgram = tf.Program(InitWaveSimulation)

# Create a program that steps the wave simulation
StepWaveSimulationProgram = tf.Program(StepWaveSimulation)

#u, v = InitWaveSimulationProgram([]) # Initialize the wave simulation
#
## Step the wave simulation 100 times
#for i in range(100):
#    u, v = StepWaveSimulationProgram([u, v])
#
## get the numpy arrays from the tensors
#u = u.numpy()
#v = v.numpy()
#
## plot the results
#plt.imshow(u)
#plt.show()

