import TensorFrost as tf

def sum(A):
    n, m = A.shape
    sum_buf = tf.zeros([m], tf.float32)
    i, j = A.indices
    tf.scatterAdd(sum_buf[j], A[i, j])
    return sum_buf

def norm(A):
    sum_buf = tf.zeros([1], tf.float32)
    ids = tf.indices(A.shape)
    tf.scatterAdd(sum_buf[0], A[ids] ** 2)
    return tf.sqrt(sum_buf)

#static size QR decomposition
def QRDecomposition():
    A = tf.input([-1, -1], tf.float32)

    m, n = A.shape
    Q = tf.zeros([m, n])
    R = tf.zeros([n, n])

    j = tf.index(0, [m])
    for i in range(QRS-1):
        R[i, i] = norm(A[j, i])
        Q[j, i] = A[j, i] / R[i, i]

        t, = tf.index_grid([i+1], [n])
        p, k = tf.index_grid([0, i+1], [m, n])
        R[i, t] = sum(Q[p, i] * A[p, k])
        A[p, k] -= Q[p, i] * R[i, k]

    R[n-1, n-1] = norm(A[j, n-1])
    Q[j, n-1] = A[j, n-1] / R[n-1, n-1]

    return [Q, R]

#dynamic size QR decomposition
def QRDecomposition():
    A = tf.input([-1, -1], tf.float32)

    m, n = A.shape
    Q = tf.zeros([m, n])
    R = tf.zeros([n, n])

    j = tf.index(0, [m])

    def loop_body(i):
        R[i, i] = norm(A[j, i])
        Q[j, i] = A[j, i] / R[i, i]

        t, = tf.index_grid([i+1], [n])
        p, k = tf.index_grid([0, i+1], [m, n])
        R[i, t] = sum(Q[p, i] * A[p, k])
        A[p, k] -= Q[p, i] * R[i, k]

    tf.loop(loop_body, 0, n, 1)

    R[n-1, n-1] = norm(A[j, n-1])
    Q[j, n-1] = A[j, n-1] / R[n-1, n-1]

    return [Q, R]


def MipMaps():
    A = tf.input([-1, -1], tf.float32)

    m, n = A.shape
    
    m1, n1, off = m, n, tf.zeros([1], tf.int32)

    #buffer for mipmaps
    M = tf.buffer([2 * m, n])
    i, j = tf.indices([m, n])
    M[i, j] = A[i, j]
    off.set(off + m)

    def loop_body(i):
        poff = off - m1
        m1.set(m1 / 2)
        n1.set(n1 / 2)
        i, j = tf.indices([m1, n1])

        #downsample
        M[i + off, j] = (M[2 * i + poff, 2 * j] + M[2 * i + poff, 2 * j + 1] + M[2 * i + poff + 1, 2 * j] + M[2 * i + poff + 1, 2 * j + 1]) / 4.0
        