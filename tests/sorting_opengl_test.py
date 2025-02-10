# %%
import numpy as np
import TensorFrost as tf
import unittest

tf.initialize(tf.opengl)

def Sort0():
    keys = tf.input([-1], tf.uint32)
    values = tf.input([-1], tf.uint32)
    sorted_keys, sorted_values = tf.sort.radix(keys, values)
    return sorted_keys, sorted_values

def Sort1():
    keys = tf.input([-1], tf.uint32)
    values = tf.input([-1], tf.uint32)
    sorted_keys, sorted_values = tf.sort.bitonic(keys, values)
    return sorted_keys, sorted_values

def Sort2():
    keys = tf.input([-1], tf.float32)
    values = tf.input([-1], tf.float32)
    sorted_keys, sorted_values = tf.sort.radix(keys, values)
    return sorted_keys, sorted_values

class TestSorting(unittest.TestCase):
    def test_sorting(self):
        #compile the program
        sort_program0 = tf.compile(Sort0)
        sort_program1 = tf.compile(Sort1)
        sort_program2 = tf.compile(Sort2)

        # Generate some random data to sort (ints between 0 and 10)
        N = 2**20
        MaxValue = 2**31 - 1
        keys = np.random.randint(0, MaxValue, N).astype(np.uint32)
        values = np.random.randint(0, MaxValue, N).astype(np.uint32)

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
        print("Radix uint errors: ", error_radix)
        print("Bitonic uint errors: ", error_bitonic)
        self.assertTrue(error_radix == 0)
        self.assertTrue(error_bitonic == 0)

        # Generate some random data to sort (floats between 0 and 1)
        keys = np.random.rand(N).astype(np.float32)
        values = np.random.rand(N).astype(np.float32)

        # do sort in TensorFrost
        sorted_keys0, sorted_values0 = sort_program2(keys, values)

        # do argsort in numpy
        sorted_indices = np.argsort(keys)
        sorted_keys2 = keys[sorted_indices]
        sorted_values2 = values[sorted_indices]

        # check if the results are the same
        error_radix = np.sum(np.abs(sorted_keys0.numpy - sorted_keys2))
        print("Radix float errors: ", error_radix)
        self.assertTrue(error_radix == 0)