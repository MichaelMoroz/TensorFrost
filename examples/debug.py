# %%
import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)

import math

def softmax(X):
    exp = tf.exp(X)
    return exp / tf.unsqueeze(tf.sum(exp))

def forward(W1, W2, W3, b1, b2, b3, X):
    L1 = tf.tanh(tf.matmul(X, W1) + b1)
    L2 = tf.tanh(tf.matmul(L1, W2) + b2)
    return softmax(tf.matmul(L2, W3) + b3)

def loss(Y, Yhat): #cross entropy loss
    return tf.sum(tf.sum( - Y * tf.log(Yhat + 1e-6) - (1.0 - Y) * tf.log(1.0 - Yhat + 1e-6))) / tf.float(math.prod(Y.shape))

def step():
    #input weights and biases
    W1 = tf.input([-1, -1], tf.float32)
    In, Hidden1 = W1.shape
    W2 = tf.input([Hidden1, -1], tf.float32)
    Hidden2, Hidden3 = W2.shape
    W3 = tf.input([Hidden3, -1], tf.float32)
    Out = W3.shape[1]
    b1 = tf.input([Hidden1], tf.float32)
    b2 = tf.input([Hidden2], tf.float32)
    b3 = tf.input([Out], tf.float32)

    #input data
    X = tf.input([-1, In], tf.float32)
    Y = tf.input([-1, Out], tf.float32)

    info = tf.input([3], tf.float32)
    offset = tf.int(info[0])
    batch_size = tf.int(info[1])
    learning_rate = info[2]

    #TODO: implement slicing instead of this crap
    i, j = tf.indices([batch_size, In])
    Xbatch = X[i + offset, j]
    i, j = tf.indices([batch_size, Out])
    Ybatch = Y[i + offset, j]

    Yhat = forward(W1, W2, W3, b1, b2, b3, Xbatch)
    L = loss(Ybatch, Yhat)

    #dW1, dW2, db1, db2 = backward(W1, W2, b1, b2, L1, Yhat, Xbatch, Ybatch)
    # dW3 = tf.grad(L, W3)
    # db3 = tf.grad(L, b3)
    # dW2 = tf.grad(L, W2)
    # db2 = tf.grad(L, b2)
    # dW1 = tf.grad(L, W1)
    # db1 = tf.grad(L, b1)

    W1 = W1 - learning_rate * tf.grad(L, W1)
    W2 = W2 - learning_rate * tf.grad(L, W2)
    W3 = W3 - learning_rate * tf.grad(L, W3)
    b1 = b1 - learning_rate * tf.grad(L, b1)
    b2 = b2 - learning_rate * tf.grad(L, b2)
    b3 = b3 - learning_rate * tf.grad(L, b3)

    return [L, W1, W2, W3, b1, b2, b3]

train_step = tf.compile(step)