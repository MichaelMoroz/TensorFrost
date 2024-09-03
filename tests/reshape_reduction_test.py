import TensorFrost as tf
import numpy as np
import unittest

tf.initialize(tf.cpu)

def ReductionReshape():
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
	return max_val, min_val, sum, mean, norm, total_max, total_min, mean_block

class ReductionReshapeTest(unittest.TestCase):
    def test_reduction_reshape(self):
        max_block = tf.compile(ReductionReshape)

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
        
        self.assertTrue(np.allclose(max_val_tf.numpy, max_val_np))
        self.assertTrue(np.allclose(min_val_tf.numpy, min_val_np))
        self.assertTrue(np.allclose(sum_tf.numpy, sum_np))
        self.assertTrue(np.allclose(mean_tf.numpy, mean_np))
        self.assertTrue(np.allclose(norm_tf.numpy, norm_np))
        self.assertTrue(np.allclose(total_max_tf.numpy, total_max_np))
        self.assertTrue(np.allclose(total_min_tf.numpy, total_min_np))
        self.assertTrue(np.allclose(mean_block_tf.numpy, mean_block_np))