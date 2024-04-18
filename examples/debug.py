import TensorFrost as tf
import numpy as np

tf.initialize(tf.opengl)

def MaxBlock():
	A = tf.input([-1, -1, -1, -1], tf.float32)
	N, Bx, By, Bz = A.shape
	Ar = tf.reshape(A, [N, Bx*By*Bz])
	#only reduces one dimension, by default it is the last dimension
	max_val = tf.max(Ar)
	min_val = tf.min(Ar)
	sum = tf.sum(Ar)
	mean = tf.mean(Ar)
	norm = tf.norm(Ar)
	total_max = tf.max(max_val)
	total_min = tf.min(min_val)
	#get mean block
	mean_block = tf.mean(A, axis=0)
	return [max_val, min_val, sum, mean, norm, total_max, total_min, mean_block]

max_block = tf.compile(MaxBlock)


#generate random tensor
A = np.random.rand(10, 5, 6, 7)

#compute maximum block using TensorFrost
Atf = tf.tensor(A)
max_val_tf, min_val_tf, sum_tf, mean_tf, norm_tf, total_max_tf, total_min_tf, mean_block_tf = max_block(Atf)
max_val_np = np.max(A, axis=(1, 2, 3))
min_val_np = np.min(A, axis=(1, 2, 3))
sum_np = np.sum(A, axis=(1, 2, 3))
mean_np = np.mean(A, axis=(1, 2, 3))
norm_np = np.linalg.norm(A.reshape(10, -1), axis=1)
total_max_np = np.max(max_val_np)
total_min_np = np.min(min_val_np)
mean_block_np = np.mean(A, axis=0)

#check if maximum block is correct
print("Maximum block using TensorFrost is correct:", np.allclose(max_val_tf.numpy, max_val_np))
print("Minimum block using TensorFrost is correct:", np.allclose(min_val_tf.numpy, min_val_np))
print("Sum block using TensorFrost is correct:", np.allclose(sum_tf.numpy, sum_np))
print("Mean block using TensorFrost is correct:", np.allclose(mean_tf.numpy, mean_np))
print("Norm block using TensorFrost is correct:", np.allclose(norm_tf.numpy, norm_np))
print("Total maximum using TensorFrost is correct:", np.allclose(total_max_tf.numpy, total_max_np))
print("Total minimum using TensorFrost is correct:", np.allclose(total_min_tf.numpy, total_min_np))
print("Mean block using TensorFrost is correct:", np.allclose(mean_block_tf.numpy, mean_block_np))

#check error
print("Error using TensorFrost:", np.linalg.norm(max_val_np - max_val_tf.numpy))

#print maximum block
print("Maximum block:", max_val_tf.numpy)
print("Maximum block:", max_val_np)

#print minimum block
print("Minimum block:", min_val_tf.numpy)
print("Minimum block:", min_val_np)

#print sum block
print("Sum block:", sum_tf.numpy)
print("Sum block:", sum_np)

#print mean block
print("Mean block:", mean_tf.numpy)
print("Mean block:", mean_np)

#print norm block
print("Norm block:", norm_tf.numpy)
print("Norm block:", norm_np)

#print total maximum
print("Total maximum:", total_max_tf.numpy)
print("Total maximum:", total_max_np)

#print total minimum
print("Total minimum:", total_min_tf.numpy)
print("Total minimum:", total_min_np)

#print mean block
print("Mean block:", mean_block_tf.numpy)
print("Mean block:", mean_block_np)