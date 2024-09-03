# %%
import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt
import unittest

tf.initialize(tf.opengl)

def PrefixSum(A, axis = -1):
    axis = len(A.shape) + axis if axis < 0 else axis
    group_size = 64
    grouped = tf.split_dim(A, group_size, axis)
    group_scan = tf.prefix_sum(tf.sum(grouped, axis = axis + 1), axis = axis)
    ids = grouped.indices
    gid, eid = ids[axis], ids[axis + 1]
    ids = [ids[i] for i in range(len(ids)) if i != axis + 1]
    ids[axis] = gid - 1
    group_scan = tf.prefix_sum(grouped + tf.select((gid == 0) | (eid != 0), 0, group_scan[tuple(ids)]), axis = axis + 1)
    full_scan = tf.merge_dim(group_scan, target_size = A.shape[axis], axis = axis + 1)
    return full_scan

def RadixSort(keys, values, bits_per_pass = 8, max_bits = 32):
    iters = (max_bits + bits_per_pass - 1) // bits_per_pass
    group_size = 128
    histogram_size = 2 ** bits_per_pass

    def GetBits(A, i):
        return (A >> (i * bits_per_pass)) & (histogram_size - 1)
    
    keys1 = tf.buffer(keys.shape, keys.type)
    values1 = tf.buffer(values.shape, values.type)
    with tf.loop(iters // 2) as iter:
        def SortIteration(A, B, C, D, iter):
            tf.region_begin('SortIteration')
            grouped = tf.split_dim(GetBits(A, iter), group_size)

            # Do a packed histogram, since we sum 128 elements at a time, we can pack 4 values into a single uint32
            g, e, i = tf.indices([grouped.shape[0], grouped.shape[1], tf.int(histogram_size/4)])
            this_key = grouped[g, e]
            packed_is_bit = (tf.uint(this_key == 4*i)) + (tf.uint(this_key == 4*i+1) << 8) + (tf.uint(this_key == 4*i+2) << 16) + (tf.uint(this_key == 4*i+3) << 24)
            packed_is_bit = tf.select((g*group_size + e) < A.shape[0], packed_is_bit, tf.uint(0))
            group_histogram_packed = tf.sum(packed_is_bit, axis = 1)
            g, i = tf.indices([grouped.shape[0], histogram_size])
            group_histogram = tf.int((group_histogram_packed[g, i / 4] >> (8*(i % 4))) & tf.uint(0xFF))

            # g, e, i = tf.indices([grouped.shape[0], grouped.shape[1], histogram_size])
            # is_bit = tf.select((grouped[g, e] == i) & ((g*group_size + e) < A.shape[0]), 1, 0)
            # group_histogram = tf.sum(is_bit, axis = 1)

            group_histogram_scan = PrefixSum(group_histogram, axis = 0)
            i, = tf.indices([histogram_size])
            total_bit_histogram = tf.prefix_sum(group_histogram_scan[group_histogram_scan.shape[0] - 1, i])
            with tf.kernel(grouped.shape) as (g, e):
                element = g * group_size + e
                with tf.if_cond(element < A.shape[0]):
                    old_key = A[element]
                    old_val = C[element]
                    bit = GetBits(old_key, iter)
                    total_offset = tf.select(g == 0, 0, group_histogram_scan[g - 1, bit]) + tf.select(bit == 0, 0, total_bit_histogram[bit - 1])
                    with tf.loop(e) as j:
                        total_offset.val += tf.int(grouped[g, j] == bit)
                    B[total_offset] = old_key
                    D[total_offset] = old_val
            tf.region_end('SortIteration')

        SortIteration(keys, keys1, values, values1, 2 * iter)
        SortIteration(keys1, keys, values1, values, 2 * iter + 1)

    return keys, values

def BitonicSort(keys, values):
    tf.region_begin('Sort')
    element_count = keys.shape[0]
    log2N = tf.ceil(tf.log2(tf.float(element_count)))
    Nround = tf.int(tf.exp2(log2N))
    sort_id = tf.indices([Nround/2])[0]
    steps = tf.int(log2N*(log2N + 1.0)/2.0)

    with tf.loop(steps) as step:
        def getBitonicElementPair(id, step):
            j = tf.floor(tf.sqrt(tf.float(2*step) + 1.0) - 0.5)
            n = tf.round(tf.float(step) - 0.5*j*(j+1.0))
            B = tf.int(tf.round(tf.exp2(j-n)))
            mask = tf.select(n < 0.5, 2*B - 1, B)
            e1 = id%B + 2*B*(id/B)
            e2 = e1 ^ mask
            return e1, e2
        e1, e2 = getBitonicElementPair(sort_id, step)

        with tf.if_cond((e1 < element_count) & (e2 < element_count)):
            key1, key2 = keys[e1], keys[e2]

            with tf.if_cond(key1 > key2):
                val1, val2 = values[e1], values[e2]
                keys[e1] = key2
                keys[e2] = key1
                values[e1] = val2
                values[e2] = val1

    tf.region_end('Sort')
    return keys, values

def Sort0():
    keys = tf.input([-1], tf.int32)
    values = tf.input([-1], tf.int32)
    sorted_keys, sorted_values = RadixSort(keys, values)
    return sorted_keys, sorted_values

def Sort1():
    keys = tf.input([-1], tf.int32)
    values = tf.input([-1], tf.int32)
    sorted_keys, sorted_values = BitonicSort(keys, values)
    return sorted_keys, sorted_values

class TestSorting(unittest.TestCase):
    def test_sorting(self):
        #compile the program
        sort_program0 = tf.compile(Sort0)
        sort_program1 = tf.compile(Sort1)

        # Generate some random data to scan (ints between 0 and 10)
        N = 2**20
        MaxValue = 2**31 - 1
        keys = np.random.randint(0, MaxValue, N).astype(np.int32)
        values = np.random.randint(0, MaxValue, N).astype(np.int32)

        # do sort in TensorFrost
        sorted_keys0, sorted_values0 = sort_program0(keys, values)
        sorted_keys1, sorted_values1 = sort_program1(keys, values)

        # do argsort in numpy
        sorted_indices = np.argsort(keys)
        sorted_keys2 = keys[sorted_indices]
        sorted_values2 = values[sorted_indices]

        # check if the results are the same
        error_radix = np.sum(np.abs(sorted_keys0.numpy - sorted_keys2))
        error_bitonic = np.sum(np.abs(sorted_keys1.numpy - sorted_keys2))
        self.assertTrue(error_radix == 0)
        self.assertTrue(error_bitonic == 0)