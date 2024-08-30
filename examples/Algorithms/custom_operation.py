import numpy as np
import TensorFrost as tf
import matplotlib.pyplot as plt

tf.initialize(tf.opengl)

def custom_op(inputs, tensor, axes):
    return [tf.tanh(inputs[0])]

def custom_op_vjp(inputs, gradient, tensor):
    return [gradient * (1.0 - tensor * tensor)]

tf.register_custom_operation("new_tanh", ["f_f"], custom_op, custom_op_vjp)

def ProgramTest():
    A = tf.input([-1],tf.float32)
    B = tf.custom("new_tanh", [A])
    dB_dA = tf.grad(B, A)
    return B, dB_dA

test = tf.compile(ProgramTest)

