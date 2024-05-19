# %%
import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)
#
# import math
#
# def softmax(X):
#     exp = tf.exp(X)
#     return exp / tf.unsqueeze(tf.sum(exp))
#
# def forward(W1, W2, W3, b1, b2, b3, X):
#     L1 = tf.tanh(tf.matmul(X, W1) + b1)
#     L2 = tf.tanh(tf.matmul(L1, W2) + b2)
#     return softmax(tf.matmul(L2, W3) + b3)
#
# def loss(Y, Yhat): #cross entropy loss
#     return tf.sum(tf.sum( - Y * tf.log(Yhat + 1e-5) - (1.0 - Y) * tf.log(1.0 - Yhat + 1e-5))) / tf.float(math.prod(Y.shape))
#
# def step():
#     #input weights and biases
#     W1 = tf.input([-1, -1], tf.float32)
#     In, Hidden1 = W1.shape
#     W2 = tf.input([Hidden1, -1], tf.float32)
#     Hidden2 = W2.shape[1]
#     W3 = tf.input([Hidden2, -1], tf.float32)
#     Out = W3.shape[1]
#     b1 = tf.input([Hidden1], tf.float32)
#     b2 = tf.input([Hidden2], tf.float32)
#     b3 = tf.input([Out], tf.float32)
#
#     #input data
#     X = tf.input([-1, In], tf.float32)
#     Y = tf.input([-1, Out], tf.float32)
#
#     info = tf.input([3], tf.float32)
#     offset = tf.int(info[0])
#     batch_size = tf.int(info[1])
#     learning_rate = info[2]
#
#     #TODO: implement slicing instead of this crap
#     i, j = tf.indices([batch_size, In])
#     Xbatch = X[i + offset, j]
#     i, j = tf.indices([batch_size, Out])
#     Ybatch = Y[i + offset, j]
#
#     Yhat = forward(W1, W2, W3, b1, b2, b3, Xbatch)
#     L = loss(Ybatch, Yhat)
#
#     W1 = W1 - learning_rate * tf.grad(L, W1)
#     W2 = W2 - learning_rate * tf.grad(L, W2)
#     W3 = W3 - learning_rate * tf.grad(L, W3)
#     b1 = b1 - learning_rate * tf.grad(L, b1)
#     b2 = b2 - learning_rate * tf.grad(L, b2)
#     b3 = b3 - learning_rate * tf.grad(L, b3)
#
#     return [L, W1, W2, W3, b1, b2, b3]
#
# train_step = tf.compile(step)


# def settest():
#     a = tf.input([1], tf.float32)
#     b = tf.input([1], tf.float32)
#
#     c = tf.const(0.0)
#     t = tf.const(0.0)
#     c.set(a)
#     c.val += b
#     c.val += 1.0
#     with tf.loop(10):
#         c.val += a
#     t.val = c
#     c.val += 2.0
#
#     with tf.loop(10):
#         c.val += b
#
#     return [c, t]
#
# test = tf.compile(settest)

def test():
    a = tf.input([256])
    b = tf.input([256])

    c = a + b

    return [c]

grad = tf.compile(test)

anp = np.random.rand(256).astype(np.float32)
bnp = np.random.rand(256).astype(np.float32)
a = tf.tensor(anp)
b = tf.tensor(bnp)

c, = grad(a, b)

print(c.numpy - (anp + bnp))
