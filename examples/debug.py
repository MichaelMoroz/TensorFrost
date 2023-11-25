import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def SomeFunction2():
    A = tf.input([-1, -1], tf.float32)
    B = tf.input([-1, -1], tf.float32)

    N, M = A.shape
    K = B.shape[1]

    i, j, k = tf.indices([N, K, M])

    C = tf.zeros([N, K])
    tf.scatterAdd(C[i, j], A[i, k] * B[k, j])

    return [C]

def SomeFunction3():
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

def PoissonSolver():
    x = tf.input([-1, -1], tf.float32)
    b = tf.input([-1, -1], tf.float32)
    n = tf.input([], tf.int32)

    i, j = x.indices

    def loop(t):
        nonlocal x
        x[i, j] = (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1] - b[i, j]) / 4.0

    tf.loop(end = n, body = loop)

    return [x]

def PoissonSolver2():
    x = tf.input([-1, -1], tf.float32)
    b = tf.input([-1, -1], tf.float32)

    i, j = x.indices

    x = (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1] - b[i, j]) / 4.0

    x = (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1] - b[i, j]) / 4.0
   
    return [x]

def PoissonStep():
    u = tf.input([-1, -1], tf.float32)
    f = tf.input([-1, -1], tf.float32)

    i,j = u.indices

    u = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - f) / 4.0
    u = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - f) / 4.0

    return [u]

def TEST():
    a = tf.input([-1], tf.float32)
    b = tf.input([-1], tf.float32)

    c = (a + b) * 2.0
   
    return [c]

def WaveEq():
    u = tf.input([16, 16], tf.float32)
    v = tf.input([16, 16], tf.float32)

    i,j = u.indices

    laplacian = u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - u[i, j] * 4.0

    print("hello")

    dt = 0.1
    v_new = v + dt*laplacian
    u_new = u + dt*v_new

    return [u_new, v_new]

def WaveEq1D():
    u = tf.input([-1], tf.float32)
    v = tf.input([-1], tf.float32)

    i = u.indices[0]

    laplacian = u[i-1] + u[i+1] - u * 2.0

    dt = 0.1
    v_new = v + dt*laplacian
    u_new = u + dt*v_new

    return [u_new, v_new]

def WaveEq1Dnp(u, v):
	laplacian = np.zeros(16)
	for i in range(16):
		laplacian[i] = u[max(i-1, 0)] + u[min(i+1, 15)] - u[i] * 2.0

	dt = 0.1
	v_new = v + dt*laplacian
	u_new = u + dt*v_new

	return [u_new, v_new]

def TEST():
    a = tf.input([-1], tf.float32)
    b = tf.input([-1], tf.float32)

    c = tf.sin((a + b) / 2.0)
   
    return [c]

N = 256

def Bilinear(tex, x, y):
    xi = tf.floor(x)
    yi = tf.floor(y)
    xf = x - xi
    yf = y - yi
    xi = tf.int(xi)
    yi = tf.int(yi)
    return (1.0-xf)*(1.0-yf)*tex[xi, yi] + xf*(1.0-yf)*tex[xi+1, yi] + (1.0-xf)*yf*tex[xi, yi+1] + xf*yf*tex[xi+1, yi+1]

def FluidTest():
    vx = tf.input([N, N], tf.float32)
    vy = tf.input([N, N], tf.float32)
    density = tf.input([N, N], tf.float32)
    pressure = tf.input([N, N], tf.float32)

    dt = 0.1

    # advect velocity
    i,j = vx.indices
    x, y = tf.float(i), tf.float(j)
    x1, y1 = x - vx*dt, y - vy*dt
    #
    vx = Bilinear(vx, x1, y1)
    vy = Bilinear(vy, x1, y1)
    density = Bilinear(density, x1, y1)
    
    # add source
    source = tf.exp(-((x-N/3.0)**2.0 + (y-N/3.0)**2.0)/10.0)
    density = density + source
    vx = vx + source

    # compute divergence
    div = vx[i+1, j] - vx[i-1, j] + vy[i, j+1] - vy[i, j-1]

    # pressure solve
    for i in range(1):
        pressure = (pressure[i-1, j] + pressure[i+1, j] + pressure[i, j-1] + pressure[i, j+1] - div) / 4.0
    
    # subtract pressure gradient
    vx = vx - (pressure[i+1, j] - pressure[i-1, j]) / 2.0
    vy = vy - (pressure[i, j+1] - pressure[i, j-1]) / 2.0

    return [vx, vy, density, pressure]

#tf.initialize(tf.cpu, "H:/tinycc/win32/tcc.exe")
#tf.initialize(tf.cpu, "C:/msys64/mingw64/bin/gcc.exe")
tf.initialize(tf.cpu, "H:/cl_compile.bat") 
fluid = tf.program(FluidTest)
#test = tf.program(WaveEq)
#poisson = tf.program(PoissonSolver2)
##test.list_operations(compact=True)
##test.kernel_c()
#Anp = np.random.rand(16, 16)
#Bnp = np.random.rand(16, 16)
##print(Anp)
##print(Bnp)
#A = tf.memory(Anp)
#B = tf.memory(Bnp)
#A, B = test(A, B)
##print(C1)
##print(C1.numpy)
#
##compare with numpy
##Cnp = WaveEq1Dnp(Anp, Bnp)[0]
##print(Cnp)
#
#print("Used memory: " + str(tf.used_memory()))
#
#start = time.time()
#
#for i in range(200):
#    A, B  = test(A, B)
#    if i % 100 == 0:
#        print("Used memory: " + str(tf.used_memory()))
#        print("Time: " + str(time.time() - start))
#        start = time.time()


VX = tf.memory(np.zeros((N, N)))
VY = tf.memory(np.zeros((N, N)))
DENSITY = tf.memory(np.zeros((N, N)))
PRESSURE = tf.memory(np.zeros((N, N)))

fluid.list_operations(compact=True)

VX,VY, DENSITY, PRESSURE = fluid(VX,VY,DENSITY,PRESSURE)

#VX,VY,DENSITY,PRESSURE = fluid(VX,VY,DENSITY,PRESSURE)

#do a few steps and measure performance by timing every 100 steps
start = time.time()

#for i in range(1000):
#    VX, = fluid(VX)
#    if i % 100 == 99:
#        print("Iterations per second: " + str(100/(time.time()-start)))
#        start = time.time()

