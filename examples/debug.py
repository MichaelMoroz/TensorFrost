import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)

def RadixSort(A, bits_per_pass = 4, max_bits = 32):
    iters = max_bits // bits_per_pass
    group_size = 128
    histogram_size = 2 ** bits_per_pass

    def GetBits(A, i):
        return (A >> (i * bits_per_pass)) & (histogram_size - 1)

    B = tf.buffer(A.shape, A.type)
    with tf.loop(iters // 2) as iter:
        def SortIteration(A, B, iter):
            grouped = tf.split_dim(GetBits(A, iter), group_size)
            g, e, i = tf.indices([grouped.shape[0], grouped.shape[1], histogram_size])
            group_histogram = tf.sum(tf.select(grouped[g, e] == i, 1, 0), axis = 1)
            group_histogram_scan = tf.prefix_sum(group_histogram, axis = 0)
            i, = tf.indices([histogram_size])
            total_bit_histogram = tf.prefix_sum(group_histogram_scan[group_histogram_scan.shape[0] - 1, i])
            with tf.kernel(grouped.shape) as (g, e):
                element = g * group_size + e
                old = A[element]
                bit = GetBits(old, iter)
                total_offset = tf.select(g == 0, 0, group_histogram_scan[g - 1, bit]) + tf.select(bit == 0, 0, total_bit_histogram[bit - 1])
                with tf.loop(e) as j:
                    total_offset.val += tf.int(grouped[g, j] == bit)
                B[total_offset] = old

        SortIteration(A, B, 2 * iter)
        SortIteration(B, A, 2 * iter + 1)

    return B

def Sort():
    data = tf.input([-1], tf.int32)
    sorted = RadixSort(data)
    return sorted

sort_program = tf.compile(Sort)