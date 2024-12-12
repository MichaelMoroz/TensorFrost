from . import TensorFrost as tf

def zeros_like(tensor):
    return tf.zeros(tensor.shape, tensor.type)

def eye(n):
    i, j = tf.indices([n, n])
    return tf.select(i == j, 1.0, 0.0)

def eye_like(tensor):
    return eye(tensor.shape[0])

def ones_like(tensor):
    return tf.ones(tensor.shape, tensor.type)