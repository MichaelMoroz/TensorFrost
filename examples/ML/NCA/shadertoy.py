
import TensorFrost as tf
import numpy as np
import time

from nca import *

tf.initialize(tf.opengl)

model = CAModel()

load_model(model, "model.npz")

import re

def dump_data(dat):
  return dat.numpy

def print_vec4(ws):
  vec = "vec4(" + ",".join(["{0:.4g}".format(w) for w in ws]) + ")"
  vec = re.sub(r"\b0\.", ".", vec)
  return vec

def print_mat4(ws):
  mat = "mat4(" + ",".join(["{0:.4g}".format(w) for w in np.transpose(ws).flatten()]) + ")"
  mat = re.sub(r"\b0\.", ".", mat)
  return mat

def GELU(X):
    return 0.5*X*(1.0 + tf.tanh(np.sqrt(2.0/np.pi) * (X + 0.044715 * (X * X * X))))

def serialize_to_shadertoy(net):
    layer1_w = dump_data(net.fc1).T
    layer1_bias = dump_data(net.fc1_bias)
    layer2_w = dump_data(net.fc2).T
    layer2_bias = dump_data(net.fc2_bias)

    #activation = "GELU"

    #activation_func = "vec4 GELU(vec4 X) { return 0.5*X*(1.0 + tanh(sqrt(2.0/3.14159265358979323846) * (X + 0.044715 * (X * X * X)))); } \n"

    activation = "LeakyReLU"

    activation_func = "vec4 LeakyReLU(vec4 X) { return (X > 0.0) ? X : 0.01*X; } \n"

    line = "void NCA("

    #inputs (4*CHANNEL_N)
    for i in range(net.channel_n):
        line += "vec4 in" + str(i) + ", "
    
    #outputs (CHANNEL_N)
    for i in range(net.channel_n // 4):
        line += "out vec4 out" + str(i)
        if i < net.channel_n // 4 - 1:
            line += ", "

    line += ") { \n"

    #first layer (4*CHANNEL_N -> hidden_size) with activation
    line += "//first layer \n"
    for row in range(net.hidden_size // 4):
        line += "vec4 h0_" + str(row) + "=" + activation + "("
        for ft in range(4 * net.channel_n // 4):
            mat = layer1_w[row*4:(row+1)*4,ft*4:(ft+1)*4]
            line += print_mat4(mat) + "*in" + str(ft) + "+"
            line += "\n   "
        bias = layer1_bias[row*4:(row+1)*4]
        line += print_vec4(bias) + "); \n"
        
    #output layer (hidden_size -> CHANNEL_N) without activation
    line += "//output layer \n"
    for row in range(net.channel_n // 4):
        line += "out" + str(row) + "="
        for col in range(net.hidden_size // 4):
            mat = layer2_w[row*4:(row+1)*4,col*4:(col+1)*4]
            line += print_mat4(mat) + "*h0_" + str(col) + "+"
            line += "\n   "
        bias = layer2_bias[row*4:(row+1)*4]
        line += print_vec4(bias) + "; \n"

    line += "} \n"

    print(activation_func)
    print(line)

serialize_to_shadertoy(model)
    




