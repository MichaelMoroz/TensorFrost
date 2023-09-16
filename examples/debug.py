import TensorFrost as tf
import numpy as np
import matplotlib.pyplot as plt

def SomeFunction(inputs):
    v = tf.zeros([128, 128])
    u = tf.zeros([128, 128])
    a = tf.sin(v)
    b = tf.cos(u)
    c = a**2 + b**2
    return [c]

# Create a program that initializes the wave simulation
SomeFunctionProgram = tf.Program(SomeFunction)

SomeFunctionProgram(list())

SomeFunctionProgram.ListGraphOperations()
