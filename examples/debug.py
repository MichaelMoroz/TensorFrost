import numpy as np
import TensorFrost as tf

tf.initialize(tf.cpu)

def Test():
    A = tf.input([-1, -1], tf.float32)
    B = tf.input(A.shape, tf.float32)

    i, j = A.indices
    for t in range(16):
        A = A + tf.sin(B[i, j])
        B = B + tf.sin(A[i, j])

    return A

test = tf.compile(Test)

class Test(tf.Module):
    def __init__(self):
        super().__init__()
        self.scales = tf.Parameter([4, 4], tf.float32, random_scale = 0.0, optimize = False)
        self.zero_point = tf.Parameter([4, 4], tf.float32, random_scale = 0.0, optimize = False)
        self.sdf = tf.Parameter([-1, -1, -1], tf.float32, optimize = False)
        self.tex = tf.Parameter([32, 32, 32, 16], tf.float32, random_scale = 0.25, random_offset = 0.5)

def TestInit():
    model = Test()
    opt = adam(model, learning_rate = 0.01)
    opt.initialize_parameters_native()
    return opt.parameters()

init = tf.compile(TestInit)

# def Test():
#     A = tf.input([-1, -1], tf.float32)
#
#     with tf.kernel(A.shape, group_size=[16]) as (i, j):
#         # i, j = tf.indices([A.shape[0], 3])
#         # A[i, j] = A[i, j] + 1.0
#
#         v = tf.group_buffer(16, tf.float32)
#         tid = i.block_thread_index(0)
#
#         v[tid] = A[i, j]
#
#         tf.group_barrier()
#
#         sum = tf.const(0.0)
#         with tf.loop(tid) as k:
#             sum.val += v[k]
#
#         A[i, j] = sum.val
#
#     return A
#
# test = tf.compile(Test)
#
# A = np.random.rand(3, 16).astype(np.float32)
#
# restf = test(A)
#
# print(A)
# print(restf.numpy)

