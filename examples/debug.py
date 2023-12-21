import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

#tf.initialize(tf.cpu, "/Zi")
tf.initialize(tf.cpu, "/O2 /fp:fast /openmp:experimental")

#def test():
#    canvas = tf.buffer([32, 32, 3], tf.float32)
#
#    i,j = tf.indices([32, 32])
#    x, y = tf.float(i), tf.float(j)
#    x, y = x/32.0, y/32.0
#
#    vx = tf.sin(2.0*3.141592*x)
#    vy = tf.sin(2.0*3.141592*y)
#    mag = 0.5*tf.sqrt(vx*vx + vy*vy)
#
#    mag = tf.clamp(mag, 0.0, 1.0)
#    canvas[i, j, 0] = (0.277 + mag * (0.105 + mag * (-0.330 + mag * (-4.634 + mag * (6.228 + mag * (4.776 - 5.435 * mag))))))
#    canvas[i, j, 1] = (0.005 + mag * (1.404 + mag * (0.214 + mag * (-5.799 + mag * (14.179 + mag * (-13.745 + 4.645 * mag))))))
#    canvas[i, j, 2] = (0.334 + mag * (1.384 + mag * (0.095 + mag * (-19.332 + mag * (56.690 + mag * (-65.353 + 26.312 * mag))))))
#
#    a, = tf.indices([16])
#    canvas[a+8, 8, 0] = 1.0
#    canvas[8, a+8, 0] = 1.0
#    canvas[a+8, 24, 0] = 1.0
#    canvas[24, a+8, 0] = 1.0
#    return [canvas]
#
#
#t1 = tf.compile(test)
#
#res, = t1()
#resnp = res.numpy
#print(resnp.shape)
#print(resnp)
#
#N = 256
#M = 512
#
#def WaveEq():
#    u = tf.input([-1,-1], tf.float32)
#    v = tf.input(u.shape, tf.float32)
#
#    i,j = u.indices
#    laplacian = u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - u[i,j] * 4.0
#    force = laplacian - 0.1 * tf.sin(2.0*np.pi*u)
#    dt = 0.1
#    v_new = v + dt*force
#    u_new = u + dt*v_new
#
#    return [u_new, v_new]
#
#wave = tf.compile(WaveEq)
#
#def transpose(A):
#    N, M = A.shape
#    i, j = tf.indices([M, N])
#    return A[j, i] * 1.0
#
#def matmul():
#    A = tf.input([-1, -1], tf.float32)
#    N, M = A.shape
#    B = tf.input([M, -1], tf.float32)
#    K = B.shape[1]
#
#    Bt = transpose(B)
#
#    C = tf.zeros([N, K])
#    i, j, k = tf.indices([N, K, M])
#    tf.scatterAdd(C[i, j], A[i, k] * Bt[j, k])
#
#    return [C]
#
#mmul = tf.compile(matmul)
#print(mmul.list_operations())
#
#def test():
#    canvas = tf.zeros([8, 8], tf.float32)
#    i, j = tf.index_grid([0, 0], [8, 8], [2, 2])
#    canvas[i, j] = 1.0
#    return [canvas]
#
#t1 = tf.compile(test)
#print(t1.list_operations(compact=False))
#
#res, = t1()
#resnp = res.numpy
#print(resnp)

#QRS = 64
#
#def modified_gram_schmidt(A):
#    """
#    Implements the Modified Gram-Schmidt orthogonalization to get the QR decomposition of matrix A.
#    A = QR
#    """
#    A = A.astype(float)  # Ensure A is of float type
#    m, n = A.shape
#    Q = np.zeros((m, n))
#    R = np.zeros((n, n))
#    
#    for i in range(n-1):
#        R[i, i] = np.linalg.norm(A[:, i])
#        Q[:, i] = A[:, i] / R[i, i]
#        R[i, i+1:n] = np.dot(Q[:, i].T, A[:, i+1:n])
#        A[:, i+1:n] -= np.outer(Q[:, i], R[i, i+1:n])
#    R[n-1, n-1] = np.linalg.norm(A[:, n-1])
#    Q[:, n-1] = A[:, n-1] / R[n-1, n-1]
#    return Q, R
#
#def sum(A):
#    n, m = A.shape
#    sum_buf = tf.zeros([m], tf.float32)
#    i, j = A.indices
#    tf.scatterAdd(sum_buf[j], A[i, j])
#    return sum_buf
#
#def norm(A):
#    A = A * 1.0
#    sum_buf = tf.zeros([1], tf.float32)
#    ids = tf.indices(A.shape)
#    tf.scatterAdd(sum_buf[0], A[ids] ** 2)
#    return tf.sqrt(sum_buf)
#
#def QRDecomposition():
#    A = tf.input([QRS, QRS], tf.float32)
#
#    m, n = A.shape
#    Q = tf.zeros([m, n])
#    R = tf.zeros([n, n])
#
#    j = tf.index(0, [m])
#    for i in range(QRS-1):
#        R[i, i] = norm(A[j, i])
#        Q[j, i] = A[j, i] / R[i, i]
#
#        #R[i, i+1:n] = np.dot(Q[:, i].T, A[:, i+1:n])
#        #A[:, i+1:n] -= np.outer(Q[:, i], R[i, i+1:n])
#
#        t, = tf.index_grid([i+1], [n])
#        p, k = tf.index_grid([0, i+1], [m, n])
#        R[i, t] = sum(Q[p, i] * A[p, k])
#        A[p, k] -= Q[p, i] * R[i, k]
#
#    R[n-1, n-1] = norm(A[j, n-1])
#    Q[j, n-1] = A[j, n-1] / R[n-1, n-1]
#
#    return [Q, R]
#
#qr = tf.compile(QRDecomposition)
##print(qr.list_operations())
#
#Anp = np.random.rand(QRS, QRS).astype(np.float32)
#Qnp, Rnp = modified_gram_schmidt(Anp)
#print(Qnp)
#print(Rnp)
#
#A = tf.tensor(Anp)
#Qtf, Rtf = qr(A)
#Qerror = np.linalg.norm(Qtf.numpy - Qnp) / np.linalg.norm(Qnp)
#Rerror = np.linalg.norm(Rtf.numpy - Rnp) / np.linalg.norm(Rnp)
#print("Q error: ", Qerror)
#print("R error: ", Rerror)
#if Qerror > 1e-5 or Rerror > 1e-5:
#	print("QR decomposition failed")
#	exit(1)

#blur_d = 16
#blur_r = blur_d * 0.25
#
#def kernel(r):
#    return np.exp(-r*r/(2*blur_r*blur_r)) / (2*np.pi*blur_r*blur_r)
#
#def blur():
#    img = tf.input([-1, -1], tf.float32)
#    
#    blur_h = tf.zeros(img.shape, tf.float32)
#    blur_v = tf.zeros(img.shape, tf.float32)
#    i, j = img.indices
#
#    #horizontal blur
#    for k in range(-blur_d, blur_d+1):
#        blur_h += img[i+k, j] * kernel(k)
#
#    #vertical blur
#    for k in range(-blur_d, blur_d+1):
#        blur_v += blur_h[i, j+k] * kernel(k)
#
#    return [blur_v]
#
#tf_blur = tf.compile(blur)
#
#input_img = np.array(plt.imread("test.png"), dtype=np.float32)[:,:,0]
#
#tf_img = tf.tensor(input_img)
#output_img, = tf_blur(tf_img)
#
#plt.imshow(output_img.numpy)

n, m, k = 128, 128, 128

def transpose(A):
    N, M = A.shape
    i, j = tf.indices([M, N])
    return A[j, i] * 1.0

def matmul():
    A = tf.input([n, m], tf.float32)
    N, M = A.shape
    B = tf.input([m, k], tf.float32)
    K = B.shape[1]
        
    Bt = transpose(B)
        
    #C = tf.zeros([N, K])
    #i, j, k = tf.indices([N, K, M])
    #tf.scatterAdd(C[i, j], A[i, k] * Bt[j, k])
        
    C = tf.buffer([N, K], tf.float32)

    s = tf.zeros([N, K], tf.float32)
    i, j = s.indices
    def loop_body(k):
        s.set(s + A[i, k] * Bt[j, k])
         
    tf.loop(loop_body, 0, m, 1)
        
    i, j = C.indices
    C[i, j] = s
        
    return [C]

loop = tf.compile(matmul)

print(loop.list_operations())

A = np.random.rand(n, m).astype(np.float32)
B = np.random.rand(m, k).astype(np.float32)

tf_A = tf.tensor(A)
tf_B = tf.tensor(B)

t1 = time.time()

for i in range(100):
    tf_C, = loop(tf_A, tf_B)

t2 = time.time()

for i in range(100):
	C = np.dot(A, B)

t3 = time.time()

dC = np.linalg.norm(C - tf_C.numpy)
print("Error: ", dC)
print("Numpy time: ", t3 - t2)
print("TensorFrost time: ", t2 - t1)


#N = 512 
#M = 512
#
#def Bilinear(tex, x, y):
#    xi, yi = tf.floor(x), tf.floor(y)
#    xf, yf = x-xi, y-yi
#    xi, yi = tf.int(xi), tf.int(yi)
#    oxf, oyf = 1.0-xf, 1.0-yf
#    return tex[xi, yi]*oxf*oyf + tex[xi+1, yi]*xf*oyf + tex[xi, yi+1]*oxf*yf + tex[xi+1, yi+1]*xf*yf
#
#def CubicHermit(x):
#    x2 = x * x
#    x3 = x2 * x
#    return [-0.5 * x3 + x2 - 0.5 * x, 1.5 * x3 - 2.5 * x2 + 1.0, -1.5 * x3 + 2.0 * x2 + 0.5 * x, 0.5 * x3 - 0.5 * x2]
#
#def CubicInterp(tex, x, y):
#    xi, yi = tf.floor(x), tf.floor(y)
#    xf, yf = x-xi, y-yi
#    xi, yi = tf.int(xi), tf.int(yi)
#
#    wx = CubicHermit(xf)
#    wy = CubicHermit(yf)
#
#    valueY = 0.0
#    for j in range(-1, 3):
#        valueX = 0.0
#        for i in range(-1, 3):
#            valueX = valueX + tex[xi + i, yi + j] * wx[i + 1]
#        valueY = valueY + valueX * wy[j + 1]
#    return valueY
#
#def EulerAdvection(vx, vy, dt):
#    i,j = vx.indices
#    x, y = tf.float(i), tf.float(j)
#    x1, y1 = x - vx*dt, y - vy*dt
#    return x1, y1
#
#def RK4Advection(vx, vy, dt):
#    i, j = vx.indices
#    x, y = tf.float(i), tf.float(j)
#
#    x1, y1 = x - vx*dt/2.0, y - vy*dt/2.0
#    vx1, vy1 = Bilinear(vx, x1, y1), Bilinear(vy, x1, y1)
#
#    x2, y2 = x - vx1*dt/2.0, y - vy1*dt/2.0
#    vx2, vy2 = Bilinear(vx, x2, y2), Bilinear(vy, x2, y2)
#
#    x3, y3 = x - vx2*dt, y - vy2*dt
#    vx3, vy3 = Bilinear(vx, x3, y3), Bilinear(vy, x3, y3)
#
#    x4, y4 = x - (vx + 2.0*vx1 + 2.0*vx2 + vx3)*dt/6.0, y - (vy + 2.0*vy1 + 2.0*vy2 + vy3)*dt/6.0
#    return x4, y4
#
#def SemiLagrange(vx, vy, pressure, density, dt):
#    x1, y1 = RK4Advection(vx, vy, dt)
#    #x1, y1 = EulerAdvection(vx, vy, dt)
#
#    vx = CubicInterp(vx, x1, y1)
#    vy = CubicInterp(vy, x1, y1)
#    #pressure = CubicInterp(pressure, x1, y1)
#    #vx = Bilinear(vx, x1, y1)
#    #vy = Bilinear(vy, x1, y1)
#    #pressure = Bilinear(pressure, x1, y1)
#    #densitylin = Bilinear(density, x1, y1)
#    #dens1 = densitylin*0.99
#    #dens2 = densitylin*1.00
#    #dens3 = tf.min(dens1, dens2)
#    #dens4 = tf.max(dens1, dens2)
#    density = Bilinear(density, x1, y1)
#
#    return [vx, vy, pressure, density]
#
#def BFECC(vx, vy, pressure, density, dt):
#    i, j = vx.indices
#    x, y = tf.float(i), tf.float(j)
#    
#    # advect backwards
#    x1, y1 = x - vx*dt, y - vy*dt
#    vx1, vy1 = Bilinear(vx, x1, y1), Bilinear(vy, x1, y1)
#    density1 = Bilinear(density, x1, y1)
#
#    # advect forwards
#    x2, y2 = x + vx*dt, y + vy*dt
#    vx2, vy2 = Bilinear(vx1, x2, y2), Bilinear(vy1, x2, y2)
#    density2 = Bilinear(density1, x2, y2)
#
#    # compute backwards forwards error correction
#    vx = vx + (vx - vx2)*0.5
#    vy = vy + (vy - vy2)*0.5
#    density = density + (density - density2)*0.5
#
#    # advect corrected backwards
#    vx3, vy3 = Bilinear(vx, x1, y1), Bilinear(vy, x1, y1)
#    density3 = Bilinear(density, x1, y1)
#
#    return [vx3, vy3, pressure, density3]
#
#def Boundary(i, j):
#    N1, M1 = i.shape
#    return 1.0 - tf.float((i < 3) | (i > N1-4) | (j < 3) | (j > M1-4))
#
#def Jacobi(pressure, div, iterations):
#    i, j = pressure.indices
#
#    edge = Boundary(i, j)
#
#    # pressure solve
#    for it in range(iterations):
#        pressure = edge * (pressure[i-1, j] + pressure[i+1, j] + pressure[i, j-1] + pressure[i, j+1] - div) / 4.0
#
#    return pressure
#
#def Restrict(field):
#    N1, M1 = field.shape
#    N2, M2 = N1/2, M1/2
#    i, j = tf.indices([N2, M2])
#    i, j = 2*i, 2*j
#    return 0.25*(field[i, j] + field[i+1, j] + field[i, j+1] + field[i+1, j+1])
#
#def Prolong(field, orig):
#    i, j = orig.indices
#    i, j = i/2, j/2
#    return orig + field[i, j]
#
#def Residual(pressure, div):
#    i, j = pressure.indices
#    return div - (pressure[i-1, j] + pressure[i+1, j] + pressure[i, j-1] + pressure[i, j+1] - 4.0*pressure)
#
#def VCycle(pressure, div):
#    pressure = Jacobi(pressure, div, 1)
#
#    res = Residual(pressure, div)
#    res = Restrict(res)
#    pressure0 = Jacobi(tf.zeros(res.shape, tf.float32), 4.0*res, 8)
#
#    res1 = Residual(pressure0, 4.0*res)
#    res1 = Restrict(res1)
#    pressure1 = Jacobi(tf.zeros(res1.shape, tf.float32), 4.0*res1, 8)
#    pressure0 = Prolong(pressure1, pressure0)
#
#    pressure = Prolong(pressure0, pressure)
#
#    pressure = Jacobi(pressure, div, 1)
#
#    return pressure
#
#def PressureSolve(pressure, div):
#    pressure = VCycle(pressure, div)
#    pressure = VCycle(pressure, div)
#    return pressure
#
#def Smoothstep(edge0, edge1, x):
#    x = (x - edge0) / (edge1 - edge0)
#    x = tf.clamp(x, 0.0, 1.0)
#    return x * x * (3.0 - 2.0 * x)
#    
#def FluidTest():
#    vx = tf.input([N, M], tf.float32)
#    vy = tf.input([N, M], tf.float32)
#    pressure = tf.input([N, M], tf.float32)
#    density = tf.input([N, M], tf.float32)
#    canvas = tf.zeros([N, M, 3], tf.float32)
#
#    dt = 1.0
#    i,j = vx.indices
#    x, y = tf.float(i), tf.float(j)
#
#    vx, vy, pressure, density = SemiLagrange(vx, vy, pressure, density, dt)
#    
#    # add source
#    source = 0.16*tf.exp(-((y-M/5.0)**2.0 + (x-2.0*N/3.0)**2.0)/100.0)
#    source = source - 0.15*tf.exp(-((y-4.0*M/5.0)**2.0 + (x-N/3.0)**2.0)/100.0)
#    vy = vy + source
#    density = density + source*source
#
#    edge = Boundary(i, j)
#    vx = vx * edge
#    vy = vy * edge
#    density = tf.clamp(density * edge, -0.5, 1.0)
#
#    # pressure solve
#    # compute divergence
#    div = (vx[i+1, j] - vx[i-1, j] + vy[i, j+1] - vy[i, j-1]) / 2.0
#    curl = tf.abs(vy[i+1, j] - vy[i-1, j] - vx[i, j+1] + vx[i, j-1]) / 2.0
#
#    pressure = PressureSolve(pressure, div)
#
#    # subtract pressure gradient
#    gradx = (pressure[i+1, j] - pressure[i-1, j])*1.0
#    grady = (pressure[i, j+1] - pressure[i, j-1])*1.0
#    vx = vx - gradx
#    vy = vy - grady
#
#    # vortex confinement
#
#    # compute gradient of curl magnitude
#    gradx = (curl[i+1, j] - curl[i-1, j])*1.0
#    grady = (curl[i, j+1] - curl[i, j-1])*1.0
#
#    # normalize gradient
#    mag = tf.sqrt(gradx*gradx + grady*grady) + 1e-5
#    gradx = gradx / mag
#    grady = grady / mag
#
#    # compute vortex force
#    vortx = -grady * curl
#    vorty = gradx * curl
#
#    # add vortex force
#    vort_scale = 0.0
#    vx = vx + vortx * dt * vort_scale
#    vy = vy + vorty * dt * vort_scale
#
#    mag = 0.5*tf.sqrt(vx*vx + vy*vy)
#    #mag = 2.5*density
#
#    mag = tf.clamp(mag, 0.0, 1.0)
#    #canvas[i, j, 0] = 255.0*Smoothstep(0.0, 0.33, mag)
#    #canvas[i, j, 1] = 255.0*Smoothstep(0.33, 0.66, mag)
#    #canvas[i, j, 2] = 255.0*Smoothstep(0.66, 1.0, mag)
#    canvas[i, j, 0] = 255.0 * (0.277 + mag * (0.105 + mag * (-0.330 + mag * (-4.634 + mag * (6.228 + mag * (4.776 - 5.435 * mag))))))
#    canvas[i, j, 1] = 255.0 * (0.005 + mag * (1.404 + mag * (0.214 + mag * (-5.799 + mag * (14.179 + mag * (-13.745 + 4.645 * mag))))))
#    canvas[i, j, 2] = 255.0 * (0.334 + mag * (1.384 + mag * (0.095 + mag * (-19.332 + mag * (56.690 + mag * (-65.353 + 26.312 * mag))))))
#
#    return [vx, vy, pressure, canvas, div, density, Residual(pressure, div)]
#
#
#fluid = tf.compile(FluidTest)
#
#VX = tf.tensor(np.zeros((N, M)))
#VY = tf.tensor(np.zeros((N, M)))
#PRESSURE = tf.tensor(np.zeros((N, M)))
#DENSITY = tf.tensor(np.zeros((N, M)))
#
#VX, VY, PRESSURE, CANVAS, DIV, DENSITY, RES = fluid(VX, VY, PRESSURE, DENSITY)