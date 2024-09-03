import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt
import unittest

tf.initialize(tf.cpu)

def Test():
    data = tf.input([-1, -1, -1], tf.int32)
    splitted = tf.split_dim(data, 32, 0)
    merged = tf.merge_dim(splitted, axis = 1)
    return merged, splitted

def test_split_dim():
    testprog = tf.compile(Test)

    data = np.random.randint(0, 100, (128, 128, 32), dtype=np.int32)
    merged, splitted = testprog(data)

    assert np.sum(np.abs(data - merged.numpy)) == 0
    assert merged.shape == (128, 128, 32)
    assert splitted.shape == (4, 128, 128, 32)

if __name__ == '__main__':
    test_case = unittest.FunctionTestCase(test_split_dim)
    unittest.main()