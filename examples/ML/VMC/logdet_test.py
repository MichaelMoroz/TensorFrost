import numpy as np
import torch
import TensorFrost as tf
import matplotlib.pyplot as plt

from logdet import *

tf.initialize(tf.opengl)

register_logdet()

def ProgramTest():
    A = tf.input([-1, -1, -1], tf.float32)
    A = tf.assert_tensor(A, [A.shape[0], A.shape[1], A.shape[1]], tf.float32)
    B = tf.custom("logdet", [A], [A.shape[0]])
    loss = tf.sum(B*B)
    dL_dA = tf.grad(loss, A)
    return B, dL_dA, loss

test = tf.compile(ProgramTest)

tf.renderdoc_start_capture()
#np.random.seed(0)
A = np.random.randn(16, 16, 16).astype(np.float32)
B, dB_dA, loss = test(A)
tf.renderdoc_end_capture()

print("logdet(A) = ", B.numpy)
# Compare with pytorch
A_torch = torch.tensor(A, requires_grad=True)
B_torch = torch.linalg.slogdet(A_torch)[1]
loss_torch = (B_torch**2).sum()
loss_torch.backward()
print("logdet(A) = ", B_torch)

print("loss = ", loss_torch.item())
print("loss = ", loss.numpy)


# gradient error
print("Gradient error = ", np.mean(np.abs(dB_dA.numpy - A_torch.grad.numpy())))

# print the gradient
#print("Gradient = ", dB_dA.numpy )



