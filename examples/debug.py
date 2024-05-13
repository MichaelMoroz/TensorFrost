# %%
import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)

# def softmax(X):
#     exp = tf.exp(X)
#     return exp / tf.unsqueeze(tf.sum(exp))
#
# def forward(W1, W2, b1, b2, X):
#     L1 = tf.tanh(tf.matmul(X, W1) + b1)
#     L3 = softmax(tf.matmul(L1, W2) + b2)
#     return L3
#
# def forward_step():
#     #input weights and biases
#     W1 = tf.input([-1, -1], tf.float32)
#     In, Hidden = W1.shape
#     W2 = tf.input([Hidden, -1], tf.float32)
#     Out = W2.shape[1]
#     b1 = tf.input([Hidden], tf.float32)
#     b2 = tf.input([Out], tf.float32)
#
#     #input data
#     X = tf.input([-1, In], tf.float32)
#
#     Yhat = forward(W1, W2, b1, b2, X)
#
#     dW1 = tf.grad(Yhat, W1)
#     db1 = tf.grad(Yhat, b1)
#     dW2 = tf.grad(Yhat, W2)
#     db2 = tf.grad(Yhat, b2)
#
#     return [Yhat, dW1, db1, dW2, db2]
#
# fwd_step = tf.compile(forward_step)


import math

def leaky_relu(X):
    return tf.select(X > 0.0, X, 0.01 * X)

def softmax(X):
    exp = tf.exp(X)
    return exp / tf.unsqueeze(tf.sum(exp))

def leaky_relu(X):
    return tf.select(X > 0.0, X, 0.01 * X)

def leaky_relu_derivative(Y):
    return tf.select(Y > 0.0, 1.0, 0.01)

def tanh(X):
    return tf.tanh(X)

def tanh_derivative(Y):
    return 1.0 - Y * Y

def activation(X):
    #return leaky_relu(X)
    return tanh(X)

def activation_derivative(Y):
    #return leaky_relu_derivative(X)
    return tanh_derivative(Y)

def forward(W1, W2, b1, b2, X):
    L1 = activation(tf.matmul(X, W1) + b1)
    L3 = softmax(tf.matmul(L1, W2) + b2)
    return L1, L3

def loss(Y, Yhat): #cross entropy loss
    return tf.sum(tf.sum( - Y * tf.log(Yhat + 1e-6) - (1.0 - Y) * tf.log(1.0 - Yhat + 1e-6))) / tf.float(math.prod(Y.shape))

def backward(W1, W2, b1, b2, L1, L3, X, Y):
    #derivative of cross entropy loss with respect to L2
    dL2 = (L3 - Y) / tf.float(math.prod(L3.shape))
    dW2 = L1.T @ dL2
    db2 = tf.sum(dL2, axis = 0)
    dL1 = (dL2 @ W2.T) * activation_derivative(L1)
    dW1 = X.T @ dL1
    db1 = tf.sum(dL1, axis = 0)
    return dW1, dW2, db1, db2

def update(W1, W2, b1, b2, dW1, dW2, db1, db2, lr):
    W1 -= lr * dW1
    W2 -= lr * dW2
    b1 -= lr * db1
    b2 -= lr * db2
    return W1, W2, b1, b2

def step():
    #input weights and biases
    W1 = tf.input([-1, -1], tf.float32)
    In, Hidden = W1.shape
    W2 = tf.input([Hidden, -1], tf.float32)
    Out = W2.shape[1]
    b1 = tf.input([Hidden], tf.float32)
    b2 = tf.input([Out], tf.float32)

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

    L1, Yhat = forward(W1, W2, b1, b2, Xbatch)
    L = loss(Ybatch, Yhat)

    #dW1, dW2, db1, db2 = backward(W1, W2, b1, b2, L1, Yhat, Xbatch, Ybatch)
    dW2 = tf.grad(L, W2)
    db2 = tf.grad(L, b2)
    dW1 = tf.grad(L, W1)
    db1 = tf.grad(L, b1)

    W1, W2, b1, b2 = update(W1, W2, b1, b2, dW1, dW2, db1, db2, learning_rate)

    return [L, W1, W2, b1, b2]

train_step = tf.compile(step)