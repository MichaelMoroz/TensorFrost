import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.initialize(tf.cpu, "/Zi")
#tf.initialize(tf.cpu, "/O2 /fp:fast /openmp:experimental")

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

#n, m, k = 128, 128, 128
#
#def transpose(A):
#    N, M = A.shape
#    i, j = tf.indices([M, N])
#    return A[j, i] * 1.0
#
#def matmul():
#    A = tf.input([n, m], tf.float32)
#    N, M = A.shape
#    B = tf.input([m, k], tf.float32)
#    K = B.shape[1]
#        
#    Bt = transpose(B)
#        
#    #C = tf.zeros([N, K])
#    #i, j, k = tf.indices([N, K, M])
#    #tf.scatterAdd(C[i, j], A[i, k] * Bt[j, k])
#        
#    C = tf.buffer([N, K])
#
#    s = tf.zeros([N, K])
#    i, j = s.indices
#    def loop_body(k):
#        s.set(s + A[i, k] * Bt[j, k])
#         
#    tf.loop(loop_body, 0, m, 1)
#        
#    C[i, j] = s
#        
#    return [C]
#
#loop = tf.compile(matmul)
#
#print(loop.list_operations())
#
#A = np.random.rand(n, m).astype(np.float32)
#B = np.random.rand(m, k).astype(np.float32)
#
#tf_A = tf.tensor(A)
#tf_B = tf.tensor(B)
#
#t1 = time.time()
#
#for i in range(100):
#    tf_C, = loop(tf_A, tf_B)
#
#t2 = time.time()
#
#for i in range(100):
#	C = np.dot(A, B)
#
#t3 = time.time()
#
#dC = np.linalg.norm(C - tf_C.numpy)
#print("Error: ", dC)
#print("Numpy time: ", t3 - t2)
#print("TensorFrost time: ", t2 - t1)

#N = 512 
#M = 512
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
## red-black Gauss-Seidel
#def RBGS(pressure, div, iterations, overrelax=1.0):
#    N1, M1 = pressure.shape
#    i0, j0 = tf.indices([N1, M1/2])
#    
#    # pressure solve
#    for it in range(iterations):
#        new_pressure = tf.buffer([N1, M1], tf.float32)
#        # red
#        i, j = i0, 2*j0 + (i0 % 2)
#        new_pressure[i,j] = pressure[i, j] * (1.0 - overrelax) + overrelax * (pressure[i-1, j] + pressure[i+1, j] + pressure[i, j-1] + pressure[i, j+1] - div[i, j]) / 4.0
#
#        # black
#        i, j = i0, 2*j0 + ((i0+1) % 2)
#        new_pressure[i,j] = pressure[i, j] * (1.0 - overrelax) + overrelax * (new_pressure[i-1, j] + new_pressure[i+1, j] + new_pressure[i, j-1] + new_pressure[i, j+1] - div[i, j]) / 4.0
#        pressure = new_pressure
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
#    pressure = RBGS(pressure, div, 1, overrelax=1.5)
#
#   #res = Residual(pressure, div)
#   #res = Restrict(res)
#   #pressure0 = RBGS(tf.zeros(res.shape), 4.0*res, 2, overrelax=1.5)
#   #
#   #res1 = Residual(pressure0, 4.0*res)
#   #res1 = Restrict(res1)
#   #pressure1 = RBGS(tf.zeros(res1.shape), 4.0*res1, 8, overrelax=1.5)
#   #pressure1 = RBGS(pressure1, 4.0*res1, 8)
#   #pressure0 = Prolong(pressure1, pressure0)
#   #
#   #pressure0 = RBGS(pressure0, 4.0*res, 2)
#   #pressure = Prolong(pressure0, pressure)
#   #
#   #pressure = RBGS(pressure, div, 1)
#
#    return pressure
#
#def PoissonTest():
#    pressure = tf.input([N, M], tf.float32)
#    div = tf.input([N, M], tf.float32)
#
#    initial_res = Residual(pressure, div)
#
#    pressure = VCycle(pressure, div)
#
#    final_res = Residual(pressure, div)
#
#    return [pressure, initial_res, final_res]
#
#
#poissonTest = tf.compile(PoissonTest)


#def Downscale():
#    A = tf.input([-1, -1], tf.float32)
#
#    m, n = A.shape
#
#    i, j = tf.indices([m/2, n/2])
#    i, j = 2*i, 2*j
#
#    B = 0.25 * (A[i, j] + A[i+1, j] + A[i, j+1] + A[i+1, j+1])
#
#    return [B]
#
#scaler = tf.compile(Downscale)
#print(scaler.list_operations())

S = 512

def transpose(A):
    N, M = A.shape
    i, j = tf.indices([M, N])
    return A[j, i] * 1.0

def matmul():
    A = tf.input([-1, -1], tf.float32)
    N, M = A.shape
    B = tf.input([M, -1], tf.float32)
    K = B.shape[1]
        
    Bt = transpose(B)
        
    #C = tf.zeros([N, K])
    #i, j, k = tf.indices([N, K, M])
    #tf.scatterAdd(C[i, j], A[i, k] * Bt[j, k])
        
    C = tf.buffer([N, K])
        
    i, j = C.indices
        
    s = tf.zeros([N, K], tf.float32)
    def loop_body(k):
        s.set(s + A[i, k] * Bt[j, k])
         
    tf.loop(loop_body, 0, M, 1)
        
    C[i, j] = s
        
    return [C]

mmul = tf.compile(matmul)

Anp = np.random.rand(64, 32).astype(np.float32)
Bnp = np.random.rand(32, 48).astype(np.float32)

A = tf.tensor(Anp)
B = tf.tensor(Bnp)
C, = mmul(A, B)

Cnp = C.numpy

#compare to numpy
Cnp2 = Anp @ Bnp

print(Cnp)
print(Cnp2)

Cerror = np.linalg.norm(Cnp - Cnp2) / np.linalg.norm(Cnp2)
print("Error:", Cerror)