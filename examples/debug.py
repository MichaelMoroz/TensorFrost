import TensorFrost as tf

tf.initialize(tf.cpu)

def sqr(x):
    return x * x
def ProgramTest():
    A = tf.input([3], tf.float32)
    l = tf.sqrt(sqr(A[0]) + sqr(A[1]) + sqr(A[2]))
    dl_dA = tf.grad(l, A)
    return dl_dA

test = tf.compile(ProgramTest)


