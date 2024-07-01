import TensorFrost as tf
import numpy as np
tf.initialize(tf.cpu)
#
#dynamic size QR decomposition
def QRDecomposition():
    A = tf.input([-1, -1], tf.float32)

    m, n = A.shape
    Q = tf.zeros([m, n])
    R = tf.zeros([n, n])
    j = tf.index(0, [m])

    with tf.loop(n-1) as i:
        R[i, i] = tf.norm(A[j, i])
        Q[j, i] = A[j, i] / R[i, i]

        p, k = tf.index_grid([0, i + 1], [m, n])
        t, = tf.index_grid([i+1], [n])
        R[i, t] = tf.sum(Q[p, i] * A[p, k], axis=0)
        A[p, k] -= Q[p, i] * R[i, k]

    R[n-1, n-1] = tf.norm(A[j, n-1])
    Q[j, n-1] = A[j, n-1] / R[n-1, n-1]

    return [Q, R]

qr = tf.compile(QRDecomposition)

#generate random matrix
A = np.random.rand(5, 5)

#compute QR decomposition using TensorFrost
Atf = tf.tensor(A)
Qtf, Rtf = qr(Atf)
Qnp = Qtf.numpy
Rnp = Rtf.numpy

#check if QR decomposition is correct
print("QR decomposition using TensorFrost is correct:", np.allclose(A, np.dot(Qnp, Rnp)))

#check error
print("Error using TensorFrost:", np.linalg.norm(A - np.dot(Qnp, Rnp)))

#print Q and R
print("Q:\n", Qnp)
print("R:\n", Rnp)

# def test():
#     a = tf.input([2, 2], tf.float32)
#     b = tf.input([2, 2], tf.float32)
#
#     c = a + b
#     d = a - b
#     e = a * b
#
#     return [c, d, e]
#
# f = tf.compile(test)
#
# a = np.array([[1, 2], [3, 4]]).astype(np.float32)
# b = np.array([[5, 6], [7, 8]]).astype(np.float32)
#
# atf = tf.tensor(a)
# btf = tf.tensor(b)
#
# c, d, e = f(atf, btf)
#
# print(c.numpy)
# print(d.numpy)
# print(e.numpy)

# def Trilinear(tex, x, y, z):
#     xi, yi, zi = tf.floor(x), tf.floor(y), tf.floor(z)
#     xf, yf, zf = x-xi, y-yi, z-zi
#     xi, yi, zi = tf.int(xi), tf.int(yi), tf.int(zi)
#     oxf, oyf, ozf = 1.0-xf, 1.0-yf, 1.0-zf
#     return tex[xi, yi, zi]*oxf*oyf*ozf + tex[xi+1, yi, zi]*xf*oyf*ozf + tex[xi, yi+1, zi]*oxf*yf*ozf + tex[xi+1, yi+1, zi]*xf*yf*ozf + tex[xi, yi, zi+1]*oxf*oyf*zf + tex[xi+1, yi, zi+1]*xf*oyf*zf + tex[xi, yi+1, zi+1]*oxf*yf*zf + tex[xi+1, yi+1, zi+1]*xf*yf*zf
#
# def InterpGrad():
#     tex = tf.input([16, 16, 16], tf.float32)
#     pos = tf.input([-1, 3], tf.float32)
#     N = pos.shape[0]
#     vals = tf.input([N], tf.float32)
#
#     i, = vals.indices
#     x, y, z = pos[i, tf.int(0)], pos[i, tf.int(1)], pos[i, tf.int(2)]
#
#     samp = Trilinear(tex, x, y, z)
#     diff = samp-vals
#     loss = tf.sum(diff*diff)
#
#     g = tf.grad(loss, tex)
#
#     return g
#
# test = tf.compile(InterpGrad)