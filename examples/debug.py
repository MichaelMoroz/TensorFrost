import TensorFrost as tf
import numpy as np
tf.initialize(tf.cpu)

# #
#dynamic size QR decomposition
def QRDecomposition(somearg: float = 5.0):
    A = tf.input([-1, -1], tf.float32)

    A = tf.assert_tensor(A, [5, 5], tf.float32)

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

EN = 16 #embedding size
CH = 3 #number of channels

ImageW = 1000
ImageH = 1000

def Sample(tex, i, j, k):
    #return tex[i, j, k]
    res = tf.clamp(tf.round(127.0*tex[i, j, k]).pass_grad(), -127.0, 127.0)
    return res / 127.0

def Bilinear(tex, x, y, ch):
    #offset each channel to avoid discontinuities
    chidx = ch/4
    offset_x = tf.float(chidx%2) / 2.0
    offset_y = tf.float(chidx/2) / 2.0
    x, y = x+offset_x, y+offset_y
    xi, yi = tf.floor(x), tf.floor(y)
    xf, yf = x-xi, y-yi
    xi, yi = tf.int(xi), tf.int(yi)

    #fake cubic interpolation
    xf = tf.smoothstep(0.0, 1.0, xf)
    yf = tf.smoothstep(0.0, 1.0, yf)

    oxf, oyf = 1.0-xf, 1.0-yf
    #return tex[xi, yi, ch]*oxf*oyf + tex[xi+1, yi, ch]*xf*oyf + tex[xi, yi+1, ch]*oxf*yf + tex[xi+1, yi+1, ch]*xf*yf
    return Sample(tex, xi, yi, ch)*oxf*oyf + Sample(tex, xi+1, yi, ch)*xf*oyf + Sample(tex, xi, yi+1, ch)*oxf*yf + Sample(tex, xi+1, yi+1, ch)*xf*yf

def mul_bias(X, W):
    ids = tf.indices(list(X.shape[:-1]) + [W.shape[-2]])
    return tf.select(ids[-1] == X.shape[-1], 0.01, X[ids]) @ W

def GELU(x):
    return 0.5*x*(1.0+tf.tanh(0.7978845608*(x+0.044715*x*x*x)))

def Decode(tex, W1, W2, x, y):
    embed = Bilinear(tex, x, y, x.indices[-1])
    #small neural network
    embed = tf.sin(mul_bias(embed, W1))
    return (mul_bias(embed, W2))

def NeuralEmbed():
    tex = tf.input([-1, -1, EN], tf.float32)
    RN = tex.shape[0]
    pos = tf.input([-1, 2], tf.float32)
    N = pos.shape[0]
    vals = tf.input([N, CH], tf.float32)
    W1 = tf.input([EN+1, -1], tf.float32)
    HiddenSize = W1.shape[1]
    W2 = tf.input([HiddenSize+1, CH], tf.float32)

    params = tf.input([-1], tf.float32)
    tex_lr = params[0]
    weight_lr = params[1]
    normalize = tf.int(params[2])

    i, j = tf.indices([N, EN])

    with tf.if_cond(normalize == 1):
        tex[tex.indices] = tex / (tf.mean(tf.mean(tf.abs(tex), axis=0), axis=0) + 1e-6)

    samp = Decode(tex, W1, W2, pos[i, 0], pos[i, 1])

    diff = samp-vals
    loss = tf.sum(tf.sum(diff*diff)) / tf.float(N)

    W1 -= weight_lr*tf.grad(loss, W1)
    W2 -= weight_lr*tf.grad(loss, W2)
    tex -= tex_lr*tf.grad(loss, tex)

    return tex, W1, W2, loss

reconstruct = tf.compile(NeuralEmbed)