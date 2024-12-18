import TensorFrost as tf

from utils import *

def qr_decomposition_batched_tensorfrost(A):
    tf.region_begin("qr_decomposition_batched_tensorfrost")
    b, m, n = A.shape
    A = tf.copy(A)
    Q = tf.zeros([b, m, n])
    R = tf.zeros([b, n, n])
    bj, j = tf.indices([b, m])
    bi, = tf.indices([b])
    with tf.loop(n-1) as i:
        R[bi, i, i] = tf.norm(A[bj, j, i])
        Q[bj, j, i] = safe_divide(A[bj, j, i], R[bj, i, i])

        bpk, p, k = tf.index_grid([0, 0, i + 1], [b, m, n])
        summed = tf.sum(Q[bpk, p, i] * A[bpk, p, k], axis=-2)
        bt, t, = summed.indices
        t = t + i + 1
        R[bt, i, t] = summed
        A[bpk, p, k] -= Q[bpk, p, i] * R[bpk, i, k]

    R[bi, n-1, n-1] = tf.norm(A[bj, j, n-1])
    Q[bj, j, n-1] = safe_divide(A[bj, j, n-1], R[bj, n-1, n-1])
    tf.region_end("qr_decomposition_batched_tensorfrost")
    return Q, R

def invert_triangular_batched_tensorfrost(matrix, lower=True):
    tf.region_begin("invert_triangular_batched_tensorfrost")
    b, n, _ = matrix.shape
    inverted = tf.zeros([b, n, n])

    if not lower: #transpose the matrix to make it lower triangular
        matrix = matrix.T

    bi, = tf.indices([b])

    inverted[bi, 0, 0] = safe_divide(1.0, matrix[bi, 0, 0])

    with tf.loop(1, n) as i:
        inverted[bi, i, i] = safe_divide(1.0, matrix[bi, i, i])
        bpk, p, k = tf.indices([b, i, i])
        bt, t, = tf.indices([b, i])
        inverted[bt, i, t] = -tf.sum(matrix[bpk, i, p] * inverted[bpk, p, k], axis=-2) / matrix[bt, i, i]

    if not lower: #transpose the matrix back
        inverted = inverted.T

    tf.region_end("invert_triangular_batched_tensorfrost")
    return inverted

def invert_matrix(matrix):
    tf.region_begin("invert_matrix")
    Q, R = qr_decomposition_batched_tensorfrost(matrix)
    R_inv = invert_triangular_batched_tensorfrost(R, lower=False)
    tf.region_end("invert_matrix")
    return R_inv @ Q.T

def logsumsign(R):
    bi, = tf.indices([R.shape[0]])
    prod = tf.const(0.0)
    sign = tf.const(1.0)
    with tf.loop(R.shape[1]) as i:
        value = R[bi, i, i]
        prod.val += safe_log(tf.abs(value))
        sign.val *= tf.sign(value)
    return prod, sign

def logdet(matrix):
    tf.region_begin("logdet")
    Q, R = qr_decomposition_batched_tensorfrost(matrix)
    abslogdet, sign = logsumsign(R)
    packed = packfloatsign(abslogdet, sign)
    tf.region_end("logdet")
    return packed

def logdet_op(inputs, tensor, axes):
    return [logdet(inputs[0])]

def logdet_op_vjp(inputs, gradient, tensor):
    return [invert_matrix(inputs[0]).T * tf.unsqueeze(tf.unsqueeze(gradient))]

def register_logdet():
    tf.register_custom_operation("logdet", ["f_f"], logdet_op, logdet_op_vjp)