import TensorFrost as tf
import numpy as np

tf.initialize(tf.cpu)

def sqr(x):
    return x * x


def ProgramTest():
    A = tf.input([3, 1], tf.float32)
    B = tf.input([1, 3], tf.float32)
    return A + B

test = tf.compile(ProgramTest)

A = np.array([[1], [2], [3]], dtype=np.float32)
B = np.array([[1, 2, 3]], dtype=np.float32)

Atf = tf.tensor(A)
Btf = tf.tensor(B)

Ctf = test(Atf, Btf)

print(Ctf.numpy)
print(A + B)



