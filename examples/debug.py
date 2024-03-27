import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.initialize(tf.cpu, "/Zi")

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

#def DownscaleVolume(vol):
#	N, M, K = vol.shape
#	
#	i, j, k = tf.indices([N/2, M/2, K/2])
#	i, j, k = 2*i, 2*j, 2*k
#	
#	B = 0.125 * (vol[i, j, k] + vol[i+1, j, k] + vol[i, j+1, k] + vol[i+1, j+1, k] + vol[i, j, k+1] + vol[i+1, j, k+1] + vol[i, j+1, k+1] + vol[i+1, j+1, k+1])
#	
#	return B
#
#
#def Downscale():
#    A = tf.input([-1, -1, -1], tf.float32)
#    
#    A = DownscaleVolume(A)
#    A = DownscaleVolume(A)
#    A = DownscaleVolume(A)
#    
#    return [A]
#
#downscaler = tf.compile(Downscale)
#
#vol = np.random.rand(128, 128, 128).astype(np.float32)
#
#A = tf.tensor(vol)
#
#res, = downscaler(A)
#
#resnp = res.numpy
#print(resnp.shape)
#
##check downscaling by taking the mean of NxNxN block
#v0 = np.mean(vol[0:8, 0:8, 0:8])
##compare to computed
#print("Computed: ", resnp[0, 0, 0], " Expected: ", v0)
##
##
#
#N = 512 
#M = 1024
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
#    #x1, y1 = RK4Advection(vx, vy, dt)
#    x1, y1 = EulerAdvection(vx, vy, dt)
#
#    vx = Bilinear(vx, x1, y1)
#    vy = Bilinear(vy, x1, y1)
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
## red-black Gauss-Seidel
#def RBGS(pressure, div, iterations):
#    N1, M1 = pressure.shape
#    i0, j0 = tf.indices([N1, M1/2])
#
#    overrelax = 1.2
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
#    #pressure = RBGS(pressure, div, 1)
#    #
#    #res = Residual(pressure, div)
#    #res = Restrict(res)
#    #pressure0 = Jacobi(tf.zeros(res.shape), 4.0*res, 4)
#    #
#    #res1 = Residual(pressure0, 4.0*res)
#    #res1 = Restrict(res1)
#    #pressure1 = Jacobi(tf.zeros(res1.shape), 4.0*res1, 8)
#    #pressure0 = Prolong(pressure1, pressure0)
#    #
#    #pressure = Prolong(pressure0, pressure)
#
#    pressure = Jacobi(pressure, div, 1)
#
#    return pressure
#
#def PressureSolve(pressure, div):
#    pressure = VCycle(pressure, div)
#   # pressure = VCycle(pressure, div)
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
#S = 512
#VS = 128 # volume size
#density_scale = 10.0
#
#def Trilinear(tex, x, y, z):
#    xi, yi, zi = tf.floor(x), tf.floor(y), tf.floor(z)
#    xf, yf, zf = x-xi, y-yi, z-zi
#    xi, yi, zi = tf.int(xi), tf.int(yi), tf.int(zi)
#    oxf, oyf, ozf = 1.0-xf, 1.0-yf, 1.0-zf
#    return tex[xi, yi, zi]*oxf*oyf*ozf + tex[xi+1, yi, zi]*xf*oyf*ozf + tex[xi, yi+1, zi]*oxf*yf*ozf + tex[xi+1, yi+1, zi]*xf*yf*ozf + tex[xi, yi, zi+1]*oxf*oyf*zf + tex[xi+1, yi, zi+1]*xf*oyf*zf + tex[xi, yi+1, zi+1]*oxf*yf*zf + tex[xi+1, yi+1, zi+1]*xf*yf*zf
#
#def ToVolumeSpace(x, y, z):
#    return (x + 1.0) * 0.5 * float(VS), (y + 1.0) * 0.5 * float(VS), (z + 1.0) * 0.5 * float(VS)
#
#def FromVolumeSpace(x, y, z):
#    return x / float(VS) * 2.0 - 1.0, y / float(VS) * 2.0 - 1.0, z / float(VS) * 2.0 - 1.0
#
#def SampleVolume(volume, x, y, z):
#    # convert to volume space [-1.0, 1.0] to [0, VS]
#    x, y, z = ToVolumeSpace(x, y, z)
#    #check if we are outside the volume
#    check = tf.float((x>0.0) & (x<VS-1.0) & (y>0.0) & (y<VS-1.0) & (z>0.0) & (z<VS-1.0))
#    return Trilinear(volume, x, y, z) * check * density_scale
#
#def MarchRay(volume, shape, camx, camy, camz, dirx, diry, dirz, dx=0.05, steps=64):
#    td = tf.zeros(shape, tf.float32)
#    density = tf.zeros(shape, tf.float32)
#    def loop_body(k):
#        px = camx + dirx * td
#        py = camy + diry * td
#        pz = camz + dirz * td
#        rho = SampleVolume(volume, px, py, pz)
#        td.set(td + dx)
#        density.set(density + rho*dx)
#    tf.loop(loop_body, 0, steps, 1)
#    return density
#
#def MarchColor(dens, colx, coly, colz, camx, camy, camz, dirx, diry, dirz, dx):
#    td = tf.zeros([S, S], tf.float32)
#    cx = tf.zeros([S, S], tf.float32)
#    cy = tf.zeros([S, S], tf.float32)
#    cz = tf.zeros([S, S], tf.float32)
#    density = tf.zeros([S, S], tf.float32)
#    def loop_body(k):
#        px = camx + dirx * td
#        py = camy + diry * td
#        pz = camz + dirz * td
#        rho = SampleVolume(dens, px, py, pz)
#        crhox = SampleVolume(colx, px, py, pz)
#        crhoy = SampleVolume(coly, px, py, pz)
#        crhoz = SampleVolume(colz, px, py, pz)
#
#        opacity = tf.exp(-density) * rho * dx
#        cx.set(cx + crhox * opacity)
#        cy.set(cy + crhoy * opacity)
#        cz.set(cz + crhoz * opacity)
#        density.set(density + rho*dx)
#        td.set(td + dx)
#
#    tf.loop(loop_body, 0, 128, 1)
#
#    return cx, cy, cz
#
#light_dir_x = -0.577
#light_dir_y = -0.577
#light_dir_z = -0.577
#
#import numpy as np
#
#def spherical_to_cartesian(r, theta, phi):
#    # Convert spherical to Cartesian coordinates
#    x = r * np.sin(phi) * np.cos(theta)
#    y = r * np.sin(phi) * np.sin(theta)
#    z = r * np.cos(phi)
#    return x, y, z
#
#def camera_axes(r, theta, phi):
#    # Camera position
#    x, y, z = spherical_to_cartesian(r, theta, phi)
#    
#    # Forward vector (normalized vector from camera position to origin)
#    forward = np.array([-x, -y, -z]) / np.linalg.norm([x, y, z])
#    
#    # Assuming Z is up
#    world_up = np.array([0, 0, 1])
#    
#    # Right vector (cross product of world up and forward vector)
#    right = np.cross(world_up, forward)
#    right /= np.linalg.norm(right)
#    
#    # Recalculate the up vector to ensure orthogonality
#    up = np.cross(forward, right)
#    up /= np.linalg.norm(up)
#    
#    return x, y, z, up, forward, right
#
#def get_camera(i, j, phi, theta):
#    u, v = tf.float(i), tf.float(j)
#    u = (u - 0.5 * S) / float(S)
#    v = (v - 0.5 * S) / float(S)
#    camx, camy, camz, up, forward, right = camera_axes(2.0, phi, theta)
#
#    dirx = forward[0] + u * right[0] + v * up[0]
#    diry = forward[1] + u * right[1] + v * up[1]
#    dirz = forward[2] + u * right[2] + v * up[2]
#
#    # normalize direction
#    dir_mag = tf.sqrt(dirx*dirx + diry*diry + dirz*dirz)
#    dirx = dirx / dir_mag
#    diry = diry / dir_mag
#    dirz = dirz / dir_mag
#
#    return camx, camy, camz, dirx, diry, dirz
#
#def volume_ray_marcher():
#    volume = tf.input([VS, VS, VS], tf.float32)
#
#    # compute volume shadows
#    i,j,k = volume.indices
#    x, y, z = FromVolumeSpace(tf.float(i), tf.float(j), tf.float(k))
#    shadow = tf.exp(-MarchRay(volume, volume.shape, x, y, z, light_dir_x, light_dir_y, light_dir_z, 0.05, 16))
#
#    canvas = tf.zeros([S, S, 3], tf.float32)
#    i, j = tf.indices([S, S])
#    
#    camx, camy, camz, dirx, diry, dirz = get_camera(i, j, 3.0, 0.4)
#    
#    cx, cy, cz = MarchColor(volume, shadow, shadow, shadow, camx, camy, camz, dirx, diry, dirz, 0.025)
#
#    canvas[i, j, 0] = cx
#    canvas[i, j, 1] = cy
#    canvas[i, j, 2] = cz
#    
#    return [canvas]
#
#raymarch = tf.compile(volume_ray_marcher)


#def DownscaleVolumes(volumes):
#	N, M, K = volumes[0].shape
#	
#	i, j, k = tf.indices([N/2, M/2, K/2])
#	i, j, k = 2*i, 2*j, 2*k
#	
#	out_volumes = ()
#	for vol in volumes:
#		B = 0.125 * (vol[i, j, k] + vol[i+1, j, k] + vol[i, j+1, k] + vol[i+1, j+1, k] + vol[i, j, k+1] + vol[i+1, j, k+1] + vol[i, j+1, k+1] + vol[i+1, j+1, k+1])
#		out_volumes += (B,)
#	
#	return out_volumes
#
#def DownscaleVelocity4x():
#	vx = tf.input([-1, -1, -1], tf.float32)
#	vy = tf.input(vx.shape, tf.float32)
#	vz = tf.input(vx.shape, tf.float32)
#
#	vx1, vy1, vz1 = DownscaleVolumes([vx, vy, vz])
#	vx2, vy2, vz2 = DownscaleVolumes([vx1, vy1, vz1])
#      
#	return [vx2, vy2, vz2]
#
#downscaler1 = tf.compile(DownscaleVelocity4x)

#block_size = 8
#
#def BlockMaxAbs(blocks, max_block_count):
#	block_max = tf.zeros([max_block_count], tf.float32)
#	b, = block_max.indices
#
#	def loop_body(it):
#		i, j, k = it%block_size, (it/block_size)%block_size, it/(block_size*block_size)
#		block_max.set(tf.max(block_max, tf.abs(blocks[b, i, j, k])))
#
#	tf.loop(loop_body, 0, block_size*block_size*block_size, 1)
#
#	block_max = block_max + 1e-7; #float(block_size*block_size*block_size)
#	return block_max
#
#def Sparsify():
#	vol = tf.input([-1, -1, -1], tf.float32)
#	N, M, K = vol.shape
#
#	BX = N / block_size
#	BY = M / block_size
#	BZ = K / block_size
#	max_block_count = BX * BY * BZ
#	
#	b, i, j, k = tf.indices([max_block_count, block_size, block_size, block_size])
#	
#	bx, by, bz = b % BX, (b / BX) % BY, b / (BX * BY)
#
#	ii, jj, kk = i + bx * block_size, j + by * block_size, k + bz * block_size
#
#	blocks = vol[ii, jj, kk]*1.0
#
#	block_max = BlockMaxAbs(blocks, max_block_count)
#
#	counter = tf.zeros([1], tf.int32)
#	block_ids = tf.buffer([max_block_count], tf.int32)
#	b, = block_ids.indices
#
#	def if_body1():
#		index = tf.scatterAddPrev(counter[0], 1)
#		block_ids[index] = b
#	
#	#todo: compute threshold based on the block variance
#	tf.if_cond(block_max[b] > 1e-3, if_body1)
#	
#	non_empty_blocks = counter[0]
#	block_pos = tf.buffer([non_empty_blocks, 3], tf.int32)
#	b, = tf.indices([non_empty_blocks])
#	
#	block_index = block_ids[b]
#	bx, by, bz = block_index % BX, (block_index / BX) % BY, block_index / (BX * BY)
#	block_pos[b, 0] = bx
#	block_pos[b, 1] = by
#	block_pos[b, 2] = bz
#
#	b, i, j, k = tf.indices([non_empty_blocks, block_size, block_size, block_size])
#
#	block_index = block_ids[b]
#	reordered_blocks1 = blocks[block_index, i, j, k]
#
#	return [reordered_blocks1, block_pos]
#
#sparsifier = tf.compile(Sparsify)
#
#vol = np.random.rand(32, 32, 32).astype(np.float32)
#
#A = tf.tensor(vol)
#
#blocks, pos = sparsifier(A)
#
#blocksnp = blocks.numpy
#posnp = pos.numpy
#
#print(blocksnp.shape)
##compare first block
#print(blocksnp[0, 0, :, :])
#print(vol[0, 0:8, 0:8])
#
#print(posnp)
#
#def Densify():
#	blocks = tf.input([-1, block_size, block_size, block_size], tf.float32)
#	block_pos = tf.input([blocks.shape[0], 3], tf.int32)
#	shape = tf.input([3], tf.int32)
#
#	volume = tf.buffer([shape[0], shape[1], shape[2]], tf.float32)
#
#	b, i, j, k = blocks.indices
#	x, y, z = block_pos[b, 0] + i, block_pos[b, 1] + j, block_pos[b, 2] + k
#	
#	volume[x, y, z] = blocks[b, i, j, k]
#	return [volume]
#
#
#densifier = tf.compile(Densify)
#
#shape = np.array([32, 32, 32], dtype=np.int32)
#
#densified, = densifier(blocks, pos, tf.tensor(shape))
#
#densifiednp = densified.numpy
#
#print(densifiednp.shape)
#print(densifiednp[0, 0, :])
#print(vol[0, 0, :])

#S = 2048
#
#def smoothstep(x, a, b):
#	t = (x - a) / (b - a)
#	t = tf.clamp(t, 0.0, 1.0)
#	return t * t * (3.0 - 2.0 * t)
#
#def mandelbrot():
#    canvas = tf.zeros([S, S, 3], tf.float32)
#    i, j = tf.indices([S, S])
#    y, x = tf.float(i), tf.float(j)
#
#    z_re = tf.zeros([S, S], tf.float32)
#    z_im = tf.zeros([S, S], tf.float32)
#    l = tf.zeros([S, S], tf.float32)
#    c_re = x * (2.0 / S) - 1.5
#    c_im = y * (2.0 / S) - 1.0
#    def loop_body(k):
#        z_re_new = z_re*z_re - z_im*z_im + c_re
#        z_im_new = 2.0*z_re*z_im + c_im
#        z_re.set(z_re_new)
#        z_im.set(z_im_new)
#        tf.if_cond((z_re*z_re + z_im*z_im) > 256.0, lambda: tf.break_loop())
#        l.set(l + 1.0)
#         
#    tf.loop(loop_body, 0, 32, 1)
#
#    color1 = [0.0, 0.0, 0.0]
#    color2 = [0.1, 0.2, 1.0]
#
#    t = smoothstep(l, 0.0, 32.0)
#    canvas[i, j, 0] = tf.lerp(color1[0], color2[0], t)
#    canvas[i, j, 1] = tf.lerp(color1[1], color2[1], t)
#    canvas[i, j, 2] = tf.lerp(color1[2], color2[2], t)
#
#    return [canvas]
#
#mand = tf.compile(mandelbrot)
#res = mand()
#resnp = res[0].numpy


#dynamic size QR decomposition
#def QRDecomposition():
#    A = tf.input([-1, -1], tf.float32)
#
#    m, n = A.shape
#    Q = tf.zeros([m, n])
#    R = tf.zeros([n, n])
#
#    j = tf.index(0, [m])
#
#    def loop_body(i):
#        norm2 = tf.zeros([1], tf.float32)
#        def loop_body1(it):
#            norm2.set(norm2 + A[it, i] ** 2)
#        tf.loop(loop_body1, 0, m, 1)
#        R[i, i] = tf.sqrt(norm2)
#        Q[j, i] = A[j, i] / R[i, i]
#        
#        t, = tf.index_grid([i+1], [n])
#        dotprod = tf.zeros(t.shape, tf.float32)
#        def loop_body2(it):
#            dotprod.set(dotprod + Q[it, i] * A[it, t])
#        tf.loop(loop_body2, 0, m, 1)
#        R[i, t] = dotprod
#        
#        p, k = tf.index_grid([0, i+1], [m, n])
#        A[p, k] -= Q[p, i] * R[i, k]
#
#        #p, k = tf.index_grid([0, i+1], [m, n])
#        #R[i, t] = (Q[p, i] * A[p, k]).sum(axis=0) #TODO: implement sum reduction
#
#    tf.loop(loop_body, 0, n-1, 1)
#
#    norm2 = tf.zeros([1], tf.float32)
#    def loop_body1(it):
#        norm2.set(norm2 + A[it, n-1] ** 2)
#    tf.loop(loop_body1, 0, m, 1)
#    R[n-1, n-1] = tf.sqrt(norm2)
#    Q[j, n-1] = A[j, n-1] / R[n-1, n-1]
#
#    return [Q, R]
#
#qr = tf.compile(QRDecomposition)
#
#def reverseBits(num, bit_count):
#    reverse_num = tf.zeros([], tf.int32)
#    def loop_body(i):
#        tf.if_cond((num & (1 << i)) != 0, lambda: reverse_num.set(reverse_num | (1 << ((bit_count - 1) - i))))
#    tf.loop(loop_body,  0, bit_count, 1)
#    return reverse_num + ((num >> bit_count) << bit_count)
#
#def getIndexPair(i, it):
#    k1 = reverseBits(2*i, it+1)
#    k2 = k1 + (1 << it)
#    return [k1, k2]
#
##in-place FFT implementation
#def FFT():
#    Signal = tf.input([-1], tf.float32)
#    N = Signal.shape[0]
#    
#    it_num = tf.int(tf.floor(tf.log2(tf.float(N))))-1
#    Re = tf.buffer([N], tf.float32)
#    Im = tf.buffer([N], tf.float32)
#
#    i, = tf.indices([N/2])
#    
#    k1, k2 = getIndexPair(i, it_num)
#    S1 = Signal[k1]
#    S2 = Signal[k2]
#    Re[2*i] = S1 + S2
#    Im[2*i] = 0.0
#    Re[2*i+1] = S1 - S2
#    Im[2*i+1] = 0.0
#    
#    def fft_iteration(it):
#        k1, k2 = getIndexPair(i, it)
#        k3 = (k1 & ((1 << it) - 1)) * (1 << (it_num - it))
#        alpha = - 2 * np.pi * tf.float(k3) / tf.float(N)
#        Re1 = Re[k2]
#        Im1 = Im[k2]
#        Re2 = Re[k1]
#        Im2 = Im[k1]
#        C = tf.cos(alpha)
#        S = tf.sin(alpha)
#        m = C * Re1 - S * Im1
#        n = S * Re1 + C * Im1
#        Re[k1] = Re2 + m
#        Im[k1] = Im2 + n
#        Re[k2] = Re2 - m
#        Im[k2] = Im2 - n
#
#    tf.loop(fft_iteration, 1, it_num+1, 1)
#
#    return [Re, Im]
#
#fft = tf.compile(FFT)
#
#block_size = 8
#
#def BlockMaxAbs(blocks, max_block_count):
#	block_max = tf.zeros([max_block_count], tf.float32)
#	b, = block_max.indices
#
#	def loop_body(it):
#		i, j, k = it%block_size, (it/block_size)%block_size, it/(block_size*block_size)
#		block_max.set(tf.max(block_max, tf.abs(blocks[b, i, j, k])))
#
#	tf.loop(loop_body, 0, block_size*block_size*block_size, 1)
#
#	block_max = block_max + 1e-7; #float(block_size*block_size*block_size)
#	return block_max
#
#def Sparsify():
#	vol = tf.input([-1, -1, -1], tf.float32)
#	N, M, K = vol.shape
#
#	BX = N / block_size
#	BY = M / block_size
#	BZ = K / block_size
#	max_block_count = BX * BY * BZ
#	
#	b, i, j, k = tf.indices([max_block_count, block_size, block_size, block_size])
#	
#	bx, by, bz = b % BX, (b / BX) % BY, b / (BX * BY)
#
#	ii, jj, kk = i + bx * block_size, j + by * block_size, k + bz * block_size
#
#	blocks = vol[ii, jj, kk]*1.0
#
#	block_max = BlockMaxAbs(blocks, max_block_count)
#
#	counter = tf.zeros([1], tf.int32)
#	block_ids = tf.buffer([max_block_count], tf.int32)
#	b, = block_ids.indices
#
#	def if_body1():
#		index = tf.scatterAddPrev(counter[0], 1)
#		block_ids[index] = b
#	
#	#todo: compute threshold based on the block variance
#	tf.if_cond(block_max[b] > 1e-3, if_body1)
#	
#	non_empty_blocks = counter[0]
#	block_pos = tf.buffer([non_empty_blocks, 3], tf.int32)
#	b, = tf.indices([non_empty_blocks])
#	
#	block_index = block_ids[b]
#	bx, by, bz = block_index % BX, (block_index / BX) % BY, block_index / (BX * BY)
#	block_pos[b, 0] = bx
#	block_pos[b, 1] = by
#	block_pos[b, 2] = bz
#
#	b, i, j, k = tf.indices([non_empty_blocks, block_size, block_size, block_size])
#
#	block_index = block_ids[b]
#	reordered_blocks1 = blocks[block_index, i, j, k]
#
#	return [reordered_blocks1, block_pos]
#
#sparsifier = tf.compile(Sparsify)

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
#    #C = tf.zeros([N, K])
#    #i, j, k = tf.indices([N, K, M])
#    #tf.scatterAdd(C[i, j], A[i, k] * Bt[j, k])
#        
#    C = tf.buffer([N, K])
#        
#    i, j = C.indices
#        
#    s = tf.zeros([N, K], tf.float32)
#    def loop_body(k):
#        s.set(s + A[i, k] * Bt[j, k])
#         
#    tf.loop(loop_body, 0, M, 1)
#        
#    C[i, j] = s
#        
#    return [C]
#
#mmul = tf.compile(matmul)
#
#Anp = np.random.rand(64, 32).astype(np.float32)
#Bnp = np.random.rand(32, 48).astype(np.float32)
#
#A = tf.tensor(Anp)
#B = tf.tensor(Bnp)
#C, = mmul(A, B)
#
#Cnp = C.numpy
#
##compare to numpy
#Cnp2 = Anp @ Bnp
#
#print(Cnp)
#print(Cnp2)
#
#Cerror = np.linalg.norm(Cnp - Cnp2) / np.linalg.norm(Cnp2)
#print("Error:", Cerror)

#def Test():
#	A = tf.input([-1, -1], tf.float32)
#	B = tf.input(A.shape, tf.float32)
#	C = A + B
#	return [C]
#
#test = tf.compile(Test)
#
#A = np.array([[1, 2], [3, 4]]).astype(np.float32)
#B = np.array([[5, 6], [7, 8]]).astype(np.float32)
#
#Atf = tf.tensor(A)
#Btf = tf.tensor(B)
#
#C, = test(Atf, Btf)
#
#Cnp = C.numpy
#
#print(Cnp)
#
#Cnp2 = A + B
#
#print(Cnp2)

#dynamic size QR decomposition
def QRDecomposition():
    A = tf.input([-1, -1], tf.float32)

    m, n = A.shape
    Q = tf.zeros([m, n])
    R = tf.zeros([n, n])

    j = tf.index(0, [m])

    def loop_body(i):
        norm2 = tf.zeros([1], tf.float32)
        def loop_body1(it):
            norm2.set(norm2 + A[it, i] ** 2)
        tf.loop(loop_body1, 0, m, 1)
        R[i, i] = tf.sqrt(norm2)
        Q[j, i] = A[j, i] / R[i, i]
        
        t, = tf.index_grid([i+1], [n])
        dotprod = tf.zeros(t.shape, tf.float32)
        def loop_body2(it):
            dotprod.set(dotprod + Q[it, i] * A[it, t])
        tf.loop(loop_body2, 0, m, 1)
        R[i, t] = dotprod
        
        p, k = tf.index_grid([0, i+1], [m, n])
        A[p, k] -= Q[p, i] * R[i, k]

        #p, k = tf.index_grid([0, i+1], [m, n])
        #R[i, t] = (Q[p, i] * A[p, k]).sum(axis=0) #TODO: implement sum reduction

    tf.loop(loop_body, 0, n-1, 1)

    norm2 = tf.zeros([1], tf.float32)
    def loop_body1(it):
        norm2.set(norm2 + A[it, n-1] ** 2)
    tf.loop(loop_body1, 0, m, 1)
    R[n-1, n-1] = tf.sqrt(norm2)
    Q[j, n-1] = A[j, n-1] / R[n-1, n-1]

    return [Q, R]

qr = tf.compile(QRDecomposition)

