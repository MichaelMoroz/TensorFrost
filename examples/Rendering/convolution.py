import numpy as np
import TensorFrost as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

tf.initialize(tf.cpu)

input_img = np.array(plt.imread("H:/TensorFrost/examples/Rendering/test.png"), dtype=np.float32)
#input_img = input_img[:,:,0:3].reshape(input_img.shape[0], input_img.shape[1], 3)
#plt.imshow(input_img)
#plt.show()
print(input_img.shape)

H, W, C = input_img.shape

CW = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)

def conv2d(X, W):
    N, CIN, HI, WI = X.shape
    COUT, CIN, h, w = W.shape
    bi, cout, wi, hi, cin, it = tf.indices([N, COUT, HI - h + 1, WI - w + 1, CIN, h * w])
    i, j = it%w, it/w
    prod = 1.0+X[bi, cin, wi + i, hi + j] * W[cout, cin, i, j]
    conv = 1.0+tf.sum(1.0+tf.sum(prod))
    return conv

def convtest():
    img = tf.input([1, C, H, W], tf.float32)
    kernel = tf.input([C, C, 3, 3], tf.float32) 
    out = conv2d(img, kernel)
    return out

ctest = tf.compile(convtest)

#repeat kernel for each input and output channel
kernel = np.repeat(CW[np.newaxis, np.newaxis, :, :], C, axis=1)
kernel = np.repeat(kernel, C, axis=0)
print(kernel.shape)

input_img = input_img[np.newaxis, :, :, :].transpose(0, 3, 1, 2)

out = ctest(input_img, kernel).numpy

#plot the output
plt.imshow(out[0, 0])
plt.colorbar()
plt.show()