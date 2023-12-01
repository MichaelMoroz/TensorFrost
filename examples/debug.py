import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

#tf.initialize(tf.cpu, "H:/cl_compile.bat /Zi")
tf.initialize(tf.cpu, "H:/cl_compile.bat /O2 /fp:fast /openmp:experimental /Zi")

def matmul():
    A = tf.input([-1, -1], tf.float32)
    B = tf.input([-1, -1], tf.float32)

    N, M = A.shape
    K = B.shape[1]
    
    C = tf.zeros([N, K])

    i, j, k = tf.indices([N, K, M])

    tf.scatterAdd(C[i, j], A[i, k] * B[k, j])

    return [C]

def add():
    A = tf.input([-1, -1])
    B = tf.input([-1, -1])
    C = A + B
    return [C]

def QRDecomposition():
    A = tf.input([-1, -1], tf.float32)

    m, n = A.shape
    Q = tf.zeros([m, n])
    R = tf.zeros([n, n])

    def loop(i):
        j = tf.index(0, [m])
        R[i, i] = tf.sum(A[j, i] ** 2)
        Q[j, i] = A[j, i] * R[i, i]

        #R[i, i+1:n] = np.dot(Q[:, i].T, A[:, i+1:n])
        #A[:, i+1:n] -= np.outer(Q[:, i], R[i, i+1:n])

        p, k = tf.indices([m, n - i - 1])
        k += i + 1
        t = tf.index(0, [n - i - 1]) + i + 1
        R[i, t] = tf.sum(Q[p, i] * A[p, k], dim=0)
        A[p, k] -= Q[p, i] * R[i, k]
       
    tf.loop(end = n, body = loop)

    return [Q, R]

def ComputeColor():
    vx = tf.input([N, N], tf.float32)

    # compute magnitude
    #mag = tf.sqrt(vx*vx + vy*vy)

    return [vx * 255.0]

N = 1024 
M = 320

def Bilinear(tex, x, y):
    xi, yi = tf.floor(x), tf.floor(y)
    xf, yf = x-xi, y-yi
    xi, yi = tf.int(xi), tf.int(yi)
    oxf, oyf = 1.0-xf, 1.0-yf
    return tex[xi, yi]*oxf*oyf + tex[xi+1, yi]*xf*oyf + tex[xi, yi+1]*oxf*yf + tex[xi+1, yi+1]*xf*yf

def CubicHermit(x):
    x2 = x * x
    x3 = x2 * x
    return [-0.5 * x3 + x2 - 0.5 * x, 1.5 * x3 - 2.5 * x2 + 1.0, -1.5 * x3 + 2.0 * x2 + 0.5 * x, 0.5 * x3 - 0.5 * x2]

def CubicInterp(tex, x, y):
    xi, yi = tf.floor(x), tf.floor(y)
    xf, yf = x-xi, y-yi
    xi, yi = tf.int(xi), tf.int(yi)

    wx = CubicHermit(xf)
    wy = CubicHermit(yf)

    valueY = 0
    for j in range(-1, 3):
        valueX = 0
        for i in range(-1, 3):
            valueX = valueX + tex[xi + i, yi + j] * wx[i + 1]
        valueY = valueY + valueX * wy[j + 1]
    return valueY

def EulerAdvection(vx, vy, dt):
    i,j = vx.indices
    x, y = tf.float(i), tf.float(j)
    x1, y1 = x - vx*dt, y - vy*dt
    return x1, y1

def RK4Advection(vx, vy, dt):
    i, j = vx.indices
    x, y = tf.float(i), tf.float(j)

    x1, y1 = x - vx*dt/2.0, y - vy*dt/2.0
    vx1, vy1 = Bilinear(vx, x1, y1), Bilinear(vy, x1, y1)

    x2, y2 = x - vx1*dt/2.0, y - vy1*dt/2.0
    vx2, vy2 = Bilinear(vx, x2, y2), Bilinear(vy, x2, y2)

    x3, y3 = x - vx2*dt, y - vy2*dt
    vx3, vy3 = Bilinear(vx, x3, y3), Bilinear(vy, x3, y3)

    x4, y4 = x - (vx + 2.0*vx1 + 2.0*vx2 + vx3)*dt/6.0, y - (vy + 2.0*vy1 + 2.0*vy2 + vy3)*dt/6.0
    return x4, y4

def SemiLagrange(vx, vy, pressure, dt):
    # advect velocity
    #x1, y1 = RK4Advection(vx, vy, dt)
    x1, y1 = EulerAdvection(vx, vy, dt)

    #vx = CubicInterp(vx, x1, y1)
    #vy = CubicInterp(vy, x1, y1)
    #pressure = CubicInterp(pressure, x1, y1)
    vx = CubicInterp(vx, x1, y1)
    vy = CubicInterp(vy, x1, y1)
    #pressure = Bilinear(pressure, x1, y1)

    return [vx, vy, pressure]

def BFECC(vx, vy, pressure, dt):
    i, j = vx.indices
    x, y = tf.float(i), tf.float(j)
    
    # advect backwards
    x1, y1 = x - vx*dt, y - vy*dt
    vx1, vy1 = Bilinear(vx, x1, y1), Bilinear(vy, x1, y1)

    # advect forwards
    x2, y2 = x + vx*dt, y + vy*dt
    vx2, vy2 = Bilinear(vx1, x2, y2), Bilinear(vy1, x2, y2)

    # compute backwards forwards error correction
    vx = vx + (vx - vx2)*0.5
    vy = vy + (vy - vy2)*0.5

    # advect corrected backwards
    vx3, vy3 = Bilinear(vx, x1, y1), Bilinear(vy, x1, y1)

    return [vx3, vy3, pressure]

def Jacobi(pressure, div):
    i, j = pressure.indices

    # pressure solve
    pressure = (pressure[i-1, j] + pressure[i+1, j] + pressure[i, j-1] + pressure[i, j+1] - div) / 4.0

    return pressure

def Smoothstep(edge0, edge1, x):
    x = (x - edge0) / (edge1 - edge0)
    x = tf.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
    
def FluidTest():
    vx = tf.input([N, M], tf.float32)
    vy = tf.input([N, M], tf.float32)
    pressure = tf.input([N, M], tf.float32)

    dt = 1.0
    i,j = vx.indices
    x, y = tf.float(i), tf.float(j)

    vx, vy, pressure = SemiLagrange(vx, vy, pressure, dt)
    
    # add source
    source = 0.26*tf.exp(-((x-N/5.0)**2.0 + (y-2.0*M/3.0)**2.0)/100.0)
    source = source - 0.25*tf.exp(-((x-4.0*N/5.0)**2.0 + (y-M/3.0)**2.0)/100.0)
    vx = vx + source

    edge = 1.0 - tf.float((i < 2) | (i > N-3) | (j < 2) | (j > M-3))
    vx = vx * edge
    vy = vy * edge

    # pressure solve
    # compute divergence
    div = (vx[i+1, j] - vx[i-1, j] + vy[i, j+1] - vy[i, j-1]) / 2.0

    # pressure solve
    for it in range(12):
        pressure = pressure * edge
        pressure = Jacobi(pressure, div)
    
    # subtract pressure gradient
    gradx = (pressure[i+1, j] - pressure[i-1, j])*1.0
    grady = (pressure[i, j+1] - pressure[i, j-1])*1.0
    vx = vx - gradx
    vy = vy - grady

    mag = 0.2*tf.sqrt(vx*vx + vy*vy)

    r, g, b = 255.0*Smoothstep(0.0, 0.33, mag), 255.0*Smoothstep(0.33, 0.66, mag), 255.0*Smoothstep(0.66, 1.0, mag)

    return [vx, vy, pressure, r, g, b]


#fluid = tf.program(FluidTest)
#fluid.list_operations(compact=False)

mmul = tf.program(matmul)
mmul.list_operations(compact=False)
mmul.kernel_c()

Anp = np.random.rand(64, 64).astype(np.float32)
Bnp = np.random.rand(64, 64).astype(np.float32)

A = tf.memory(np.transpose(Anp))
B = tf.memory(np.transpose(Bnp))
C, = mmul(A, B)

Cnp = C.numpy

print(Cnp)

#compare to numpy
Cnp2 = np.dot(Bnp, Anp)
print(Cnp2)

print(Cnp - Cnp2)

