import numpy as np
import TensorFrost as tf

tf.initialize(tf.cpu)

def Test():
    A = tf.input([-1], tf.float32)

    Asum = tf.sum(A)

    value = Asum[0]
    tf.print_value("sum result", value)
    tf.assert_value("sum is not 15.0!", value == 15.0)

    return Asum

test = tf.compile(Test)

A = np.array([1, 2, 3, 4, 6], dtype=np.float32)

restf = test(A)

