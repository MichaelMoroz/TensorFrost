import TensorFrost as tf
import numpy as np

def dowscale_test():
    def downscale2x(a):
        N, M = a.shape
        i, j = tf.indices([N//2, M//2])
        return (a[i*2, j*2] + a[i*2+1, j*2] + a[i*2, j*2+1] + a[i*2+1, j*2+1]) * 0.25

    def tensorfrost():
        a = tf.input([-1, -1], tf.float32)
        
        for i in range(4):
            a = downscale2x(a)

        return [a]
    
    def numpy(a):
        for i in range(4):
            a = (a[::2,::2] + a[1::2,::2] + a[::2,1::2] + a[1::2,1::2]) * 0.25
        return a
    
    tf_downscale = tf.compile(tensorfrost)
    
    #create random input of random size
    N = np.random.randint(100, 200)
    M = np.random.randint(100, 200)
    a = np.random.rand(N, M).astype(np.float32)

    #run both implementations
    res = tf_downscale(a)
    resnp = numpy(a)

    #compare results
    assert np.allclose(res, resnp, atol=1e-5)

def matrix_mul_test():
    def tensorfrost():
        A = tf.input([-1, -1], tf.float32)
        N, M = A.shape
        B = tf.input([M, -1], tf.float32)
        K = B.shape[1]

        C = tf.zeros([N, K])
        i, j, k = tf.indices([N, K, M])
        tf.scatterAdd(C[i, j], A[i, k] * B[k, j])

        return [C]

    def numpy(A, B):
        return np.matmul(A, B)

    tf_mmul = tf.compile(tensorfrost)
    
    #create random input of random size
    N = np.random.randint(100, 200)
    M = np.random.randint(100, 200)
    K = np.random.randint(100, 200)
    A = np.random.rand(N, M).astype(np.float32)
    B = np.random.rand(M, K).astype(np.float32)

    #run both implementations
    res = tf_mmul(A, B)
    resnp = numpy(A, B)

    #compare results
    assert np.allclose(res, resnp, atol=1e-5)

def gaussian_blur_test():
    RBlur = 16

    def kernel(i, j):
        i, j = tf.float(i), tf.float(j)
        return tf.exp(-(i*i+j*j)/(2*RBlur*RBlur)) / (2*np.pi*RBlur*RBlur)

    def tensorfrost():
        img = tf.input([-1, -1], tf.float32)
        
        blur = tf.zeros(img.shape, tf.float32)
        i, j = img.indices

        #horizontal blur
        for k in range(-RBlur, RBlur+1):
            blur += img[i+k, j] * kernel(k, 0)

        #vertical blur
        for k in range(-RBlur, RBlur+1):
            blur += img[i, j+k] * kernel(0, k)

        return [blur]
    
    def np_kernel(i, j):
        return np.exp(-(i*i+j*j)/(2*RBlur*RBlur)) / (2*np.pi*RBlur*RBlur)

    def np_sample(img, i, j):
        return img[np.clip(i, 0, img.shape[0]-1), np.clip(j, 0, img.shape[1]-1)]

    def numpy(img):
        blur = np.zeros(img.shape, np.float32)
        
        #horizontal blur
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(-RBlur, RBlur+1):
                    blur[i, j] += np_sample(img, i+k, j) * np_kernel(k, 0)

        #vertical blur
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(-RBlur, RBlur+1):
                    blur[i, j] += np_sample(img, i, j+k) * np_kernel(0, k)

        return blur
    
    tf_blur = tf.compile(tensorfrost)

    #create random input of random size
    N = np.random.randint(64, 128)
    M = np.random.randint(64, 128)
    img = np.random.rand(N, M).astype(np.float32)

    #run both implementations
    res = tf_blur(img)
    resnp = numpy(img)

    #compare results
    assert np.allclose(res, resnp, atol=1e-5)

def qr_decomposition_test():
    #get random shape for matrix
    QRS = np.random.randint(16, 32)

    def sum(A):
        n, m = A.shape
        sum_buf = tf.zeros([m], tf.float32)
        i, j = A.indices
        tf.scatterAdd(sum_buf[j], A[i, j])
        return sum_buf

    def norm(A):
        A = A * 1.0
        sum_buf = tf.zeros([1], tf.float32)
        ids = tf.indices(A.shape)
        tf.scatterAdd(sum_buf[0], A[ids] ** 2)
        return tf.sqrt(sum_buf)

    def tensorfrost():
        A = tf.input([QRS, QRS], tf.float32)

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

    def numpy(A):
        Q, R = np.linalg.qr(A)
        return Q, R

    tf_qr = tf.compile(tensorfrost)

    #create random input
    A = np.random.rand(QRS, QRS).astype(np.float32)

    #run both implementations
    Q, R = tf_qr(A)
    Qnp, Rnp = numpy(A)

    #compare results
    assert np.allclose(Q, Qnp, atol=1e-5)
    assert np.allclose(R, Rnp, atol=1e-5)