import TensorFrost as tf
import numpy as np

# Create a 2x3 tensor
tensor = tf.Tensor([2, 2]) + [[4, 5], [6, 7]]

print(tensor.shape)
print(tensor.type)
