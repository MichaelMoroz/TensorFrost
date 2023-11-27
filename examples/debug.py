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

N = 512

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

def SemiLagrange(vx, vy, density, pressure, dt):
    # advect velocity
    x1, y1 = EulerAdvection(vx, vy, dt)

    vx = Bilinear(vx, x1, y1)
    vy = Bilinear(vy, x1, y1)
    density = Bilinear(density, x1, y1)
    #pressure = CubicInterp(pressure, x1, y1)

    return [vx, vy, density, pressure]

def BFECC(vx, vy, density, dt):
    i, j = vx.indices
    x, y = tf.float(i), tf.float(j)
    
    # advect backwards
    x1, y1 = x - vx*dt, y - vy*dt
    vx1, vy1, density1 = Bilinear(vx, x1, y1), Bilinear(vy, x1, y1), Bilinear(density, x1, y1)

    # advect forwards
    x2, y2 = x + vx*dt, y + vy*dt
    vx2, vy2, density2 = Bilinear(vx1, x2, y2), Bilinear(vy1, x2, y2), Bilinear(density1, x2, y2)

    # compute backwards forwards error correction
    vx = vx + (vx - vx2)*0.5
    vy = vy + (vy - vy2)*0.5
    density = density + (density - density2)*0.5

    # advect corrected backwards
    vx3, vy3, density3 = Bilinear(vx, x1, y1), Bilinear(vy, x1, y1), Bilinear(density, x1, y1)

    return [vx3, vy3, density3]

def Jacobi(pressure, div):
    i, j = pressure.indices

    # pressure solve
    pressure = (pressure[i-1, j] + pressure[i+1, j] + pressure[i, j-1] + pressure[i, j+1] - div) / 4.0

    return pressure

def PressureSolver(vx, vy, pressure, edge):
    i, j = vx.indices

    # compute divergence
    div = (vx[i+1, j] - vx[i-1, j] + vy[i, j+1] - vy[i, j-1]) / 2.0

    ## downsample pressure
    #i2, j2 = tf.indices([N//2, N//2])
    #i2, j2 = i2*2, j2*2
    #pressure2x = 0.25 * (pressure[i2, j2] + pressure[i2+1, j2] + pressure[i2, j2+1] + pressure[i2+1, j2+1])
    #div2x = 0.25 * (div[i2, j2] + div[i2+1, j2] + div[i2, j2+1] + div[i2+1, j2+1])
    ## pressure solve on downsampled grid
    #for it in range(4):
    #    pressure2x = Jacobi(pressure2x, div2x)

    # pressure solve
    for it in range(5):
        pressure = pressure * edge
        pressure = Jacobi(pressure, div)
    
    # subtract pressure gradient
    gradx = (pressure[i+1, j] - pressure[i-1, j])*1.0
    grady = (pressure[i, j+1] - pressure[i, j-1])*1.0
    vx = vx - gradx
    vy = vy - grady

    return [vx, vy, pressure]

def FluidTest():
    vx = tf.input([N, N], tf.float32)
    vy = tf.input([N, N], tf.float32)
    density = tf.input([N, N], tf.float32)
    pressure = tf.input([N, N], tf.float32)

    dt = 1.0
    i,j = vx.indices
    x, y = tf.float(i), tf.float(j)

    vx, vy, density = BFECC(vx, vy, density, dt)
    
    # add source
    source = 0.042*tf.exp(-((x-N/5.0)**2.0 + (y-2.0*N/3.0)**2.0)/100.0)
    source = source + 0.04*tf.exp(-((x-N/5.0)**2.0 + (y-N/3.0)**2.0)/100.0)
    density = density + source
    vx = vx + source

    edge = 1.0 - tf.float((i < 2) | (i > N-3) | (j < 2) | (j > N-3))
    vx = vx * edge
    vy = vy * edge
    density = density * edge

    # pressure solve
    # compute divergence
    div = (vx[i+1, j] - vx[i-1, j] + vy[i, j+1] - vy[i, j-1]) / 2.0

    # pressure solve
    for it in range(1):
        pressure = pressure * edge
        pressure = (pressure[i-1, j] + pressure[i+1, j] + pressure[i, j-1] + pressure[i, j+1] - div) / 4.0
    
    # subtract pressure gradient
    gradx = (pressure[i+1, j] - pressure[i-1, j])*1.0
    grady = (pressure[i, j+1] - pressure[i, j-1])*1.0
    vx = vx - gradx
    vy = vy - grady

    return [vx, vy, density, pressure]

def ComputeColor():
    vx = tf.input([N, N], tf.float32)

    # compute magnitude
    #mag = tf.sqrt(vx*vx + vy*vy)

    return [vx * 255.0]


tf.initialize(tf.cpu, "H:/cl_compile.bat /O2 /fp:fast /openmp:experimental /Zi")
#fluid = tf.program(FluidTest)
#fluid_color = tf.program(ComputeColor)
#fluid_color.list_operations(compact=False)
#tf.initialize(tf.cpu, "H:/tinycc/win32/tcc.exe")
#tf.initialize(tf.cpu, "C:/msys64/mingw64/bin/gcc.exe")
fluid = tf.program(FluidTest)
#fluid.list_operations(compact=False)
fluid.kernel_c()
#test = tf.program(WaveEq)
#poisson = tf.program(PoissonSolver2)
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


#VX = tf.memory(np.zeros((N, N)))
#VY = tf.memory(np.zeros((N, N)))
#DENSITY = tf.memory(np.zeros((N, N)))
#PRESSURE = tf.memory(np.zeros((N, N)))
#
#VX,VY, DENSITY, PRESSURE = fluid(VX,VY,DENSITY,PRESSURE)
#
##do a few steps and measure performance by timing every 100 steps
#start = time.time()

#for i in range(1000):
#    VX, = fluid(VX)
#    if i % 100 == 99:
#        print("Iterations per second: " + str(100/(time.time()-start)))
#        start = time.time()

