import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)

WX = 4
WY = 5
WZ = 6

def index_test():
    idx, = tf.indices([WX*WY*WZ])
    return tf.indices_from_flat_index(idx, [WX, WY, WZ])

test = tf.compile(index_test)

i, j, k = test()
print(i.numpy)
print(j.numpy)
print(k.numpy)