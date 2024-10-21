import numpy as np
import TensorFrost as tf

tf.initialize(tf.opengl)

def Test():
    A = tf.input([-1, -1], tf.float32)

    with tf.kernel(A.shape, group_size=[16]) as (i, j):
        # i, j = tf.indices([A.shape[0], 3])
        # A[i, j] = A[i, j] + 1.0

        v = tf.group_buffer(16, tf.float32)
        tid = i.block_thread_index(0)

        v[tid] = A[i, j]

        tf.group_barrier()

        sum = tf.const(0.0)
        with tf.loop(tid) as k:
            sum.val += v[k]

        A[i, j] = sum.val

    return A

test = tf.compile(Test)

A = np.random.rand(3, 16).astype(np.float32)

restf = test(A)

print(A)
print(restf.numpy)