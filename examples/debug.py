import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.cpu)

def Test():
    data = tf.input([-1, -1, -1], tf.int32)
    splitted = tf.split_dim(data, 32, 0)
    merged = tf.merge_dim(splitted, axis = 1)
    return merged, splitted

testprog = tf.compile(Test)

data = np.random.randint(0, 100, (128, 128, 32), dtype=np.int32)
merged, splitted = testprog(data)


print(merged.shape)
print(splitted.shape)
print(np.sum(np.abs(data - merged.numpy)))

