import TensorFrost as tf
import numpy as np
import time

tf.initialize(tf.cpu)

N = 8
def logDet():
    A = tf.input([N, N], tf.float32)

    m, n = A.shape
    Q = tf.zeros([m, n])
    R = tf.zeros([n, n])
    j = tf.index(0, [m])

    for i in range(N-1):
        R[i, i] = tf.norm(A[j, i])
        Q[j, i] = A[j, i] / R[i, i]

        p, k = tf.index_grid([0, i + 1], [m, n])
        t, = tf.index_grid([i+1], [n])
        R[i, t] = tf.sum(Q[p, i] * A[p, k], axis=0)
        A[p, k] -= Q[p, i] * R[i, k]

    R[n-1, n-1] = tf.norm(A[j, n-1])
    Q[j, n-1] = A[j, n-1] / R[n-1, n-1]

    i, = tf.indices([N])
    logDetR = tf.sum(tf.log(R[i, i]))
    dlogDetR_dA = tf.grad(logDetR, A)

    return [logDetR, dlogDetR_dA]

logDet = tf.compile(logDet)

A = np.random.randn(N, N).astype(np.float32)

logDetR, dlogDetR_dA = logDet(A)

print("logDetR: ", logDetR.numpy)
print("dlogDetR_dA: ", dlogDetR_dA.numpy)

#compare to pytorch
import torch
import torch.distributions as dist

A_torch = torch.tensor(A, requires_grad=True)
L = torch.linalg.qr(A_torch)
R_torch = L.R
logDetR_torch = torch.linalg.det(R_torch).abs().log()
logDetR_torch.backward()
dlogDetR_dA_torch = A_torch.grad

print("logDetR_torch: ", logDetR_torch.item())
print("dlogDetR_dA_torch: ", dlogDetR_dA_torch.numpy())


# compare results
diff_logDetR = np.abs(logDetR.numpy - logDetR_torch.item())
diff_dlogDetR_dA = np.abs(dlogDetR_dA.numpy - dlogDetR_dA_torch.numpy())

print("diff_logDetR: ", diff_logDetR)
print("diff_dlogDetR_dA: ", diff_dlogDetR_dA)


def InplaceTest():
    A = tf.input([N, N], tf.float32)

    i,j = A.indices()
    A[i, j] = 2.0 * A[i, j] + 1.0
    A[i, j] = 2.0 * A[i, j] + 1.0
    A[i, j] = 2.0 * A[i, j] + 1.0

    dA_dA = tf.grad(A, A) #?????






